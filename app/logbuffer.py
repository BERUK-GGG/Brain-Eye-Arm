"""Keeps the most recent printed lines so the web HMI can show them.

Every module here reports through plain print(), so rather than
switching them all to logging, main.py replaces sys.stdout with a
LogBuffer: output still reaches the terminal unchanged, and each
complete line is also kept (numbered, timestamped) for GET /logs.
"""

import io
import threading
import time
from collections import deque
from typing import List


class LogBuffer(io.TextIOBase):
    def __init__(self, stream, max_lines: int = 500):
        self._stream = stream
        self._lines = deque(maxlen=max_lines)
        self._partial = ""      # text after the last newline, not yet a line
        self._seq = 0
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        self._stream.write(text)
        with self._lock:
            *lines, self._partial = (self._partial + text).split("\n")
            for line in lines:
                if line.strip():
                    self._seq += 1
                    self._lines.append({"seq": self._seq, "time": time.strftime("%H:%M:%S"), "text": line})
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def since(self, seq: int) -> List[dict]:
        """Lines numbered after seq, oldest first. Lines that already fell
        out of the buffer are simply gone."""
        with self._lock:
            return [line for line in self._lines if line["seq"] > seq]
