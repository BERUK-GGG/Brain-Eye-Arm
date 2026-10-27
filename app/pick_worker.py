"""The robot communication thread: runs pick-and-place jobs one at a time.

Web requests only hand a job over (submit) and return at once; this
thread alone drives the robot. That gives server communication its own
thread, separate from the GUI's request threads, as the project brief
asks ("each thread responsible for at least one distinct functionality"),
and keeps the web server responsive during a cycle. It costs no
performance: the thread spends nearly all its time blocked on the robot
socket, which uses no CPU.

It also keeps the record of what it did (the HMI's pick history and the
last job's outcome), since that is exactly the outcome of its own work.
"""

import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.selection import find_same_object
from app.service import VisionService
from robot.picker import PickAndPlaceController
from vision.shapes import DetectedShape

HISTORY_LENGTH = 500  # handled objects kept for the pick history (this run only)


class WorkerBusy(RuntimeError):
    """A job is already queued or running; the arm only takes one at a time."""


@dataclass
class PickJob:
    job_id: int
    request: str                        # what the user asked for, e.g. "red Triangle"
    parts: Dict[int, DetectedShape]     # tracker ID -> part, as seen when the user asked


class PickWorker:
    """One job at a time: submit() refuses while a job is queued or running,
    so the arm only ever moves on a fresh user command (the brief requires
    user input to trigger each cycle)."""

    def __init__(self, service: VisionService, controller: PickAndPlaceController):
        self._service = service
        self._controller = controller
        self._jobs: "queue.Queue[PickJob]" = queue.Queue(maxsize=1)
        self._job_ids = itertools.count(1)
        self._submit_lock = threading.Lock()   # makes "check busy, then queue" atomic
        self._active = threading.Event()       # set from submit() until the job's outcome is recorded
        self._current: Optional[PickJob] = None

        self._record_lock = threading.Lock()   # guards the three fields below
        self._history = deque(maxlen=HISTORY_LENGTH)
        self._entry_numbers = itertools.count(1)
        self._last_outcome: Optional[dict] = None

    def start(self) -> None:
        threading.Thread(target=self._run, name="robot-worker", daemon=True).start()

    # ---------- used by the web layer ----------

    @property
    def busy(self) -> bool:
        return self._active.is_set()

    @property
    def current_request(self) -> Optional[str]:
        job = self._current
        return job.request if job else None

    def submit(self, request: str, parts: Dict[int, DetectedShape]) -> int:
        """Queues one pick cycle and returns its job ID. Raises WorkerBusy
        if a cycle is already queued or running."""
        with self._submit_lock:
            if self._active.is_set():
                raise WorkerBusy("A pick cycle is already in progress")
            self._active.set()
            job = PickJob(next(self._job_ids), request, dict(parts))
            self._jobs.put_nowait(job)
        print(f"Pick {request}: queued as job {job.job_id}, {len(parts)} part(s)")
        return job.job_id

    def history_since(self, seq: int) -> List[dict]:
        """History entries numbered after seq, oldest first."""
        with self._record_lock:
            return [entry for entry in self._history if entry["seq"] > seq]

    def last_outcome(self) -> Optional[dict]:
        with self._record_lock:
            return dict(self._last_outcome) if self._last_outcome else None

    # ---------- the robot thread ----------

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            self._current = job
            try:
                outcome = self._execute(job)
                with self._record_lock:
                    self._last_outcome = {"job": job.job_id, "time": time.strftime("%H:%M:%S"),
                                          "request": job.request, **outcome}
            finally:
                # Only now is the worker free: busy == False guarantees the
                # outcome above is already visible to /status.
                self._current = None
                self._active.clear()

    def _execute(self, job: PickJob) -> dict:
        # No tracker updates while the arm is in view -- see VisionService.
        self._service.pause()
        try:
            results = self._controller.run_cycle(
                job.parts, on_placed=self._service.discard, on_result=self._record,
                find_again=self._find_again)
        except Exception as e:  # robot link errors etc. -- the thread must survive them
            print(f"ERROR: pick {job.request} failed: {e}")
            self._record({"status": "failed", "request": job.request, "error": str(e)})
            return {"ok": False, "error": str(e)}
        finally:
            self._service.resume()
        print(f"Pick {job.request}: done -- " + ", ".join(f"{r['shape']} {r['status']}" for r in results))
        return {"ok": True, "results": results}

    def _find_again(self, shape: DetectedShape) -> Optional[DetectedShape]:
        """Is the part still lying where it was? (the arm is home, out of view)"""
        return find_same_object(shape, self._service.redetect())

    def _record(self, entry: dict) -> None:
        with self._record_lock:
            self._history.append({"seq": next(self._entry_numbers), "time": time.strftime("%H:%M:%S"), **entry})
