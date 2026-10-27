"""YOLO inference directly on ONNX Runtime.

Replaces ultralytics + torch at runtime: on the Pi those cost seconds of
startup and hundreds of MB of RAM, and ultralytics still pushed every
frame through torch tensors around the ONNX call. Only the exported
.onnx file is needed here (training stays in the notebook).

Pre/post-processing mirrors ultralytics' own, so results match what
`YOLO(model).predict()` returned for the same image:
  - letterbox to the model's input size (aspect kept, gray 114 padding)
  - decode [1, 4 + classes, anchors] -> boxes (cx, cy, w, h in input px)
  - per-class NMS at IoU 0.7, then map boxes back to image pixels
"""

import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

PAD_VALUE = 114
MAX_DETECTIONS = 300


@dataclass(frozen=True)
class Box:
    class_id: int
    confidence: float
    xyxy: Tuple[float, float, float, float]   # image pixels


class OnnxYolo:
    """One loaded YOLO .onnx model. Build once and reuse."""

    def __init__(self, path: str, threads: int = 0, iou_threshold: float = 0.7):
        """threads: ONNX Runtime intra-op threads; 0 = its default, one per
        physical core. See Detector.from_config for the value actually used."""
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self._iou_threshold = iou_threshold

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._input_h, self._input_w = model_input.shape[2], model_input.shape[3]

        # ultralytics stores the class names as a dict literal in the metadata.
        metadata = self._session.get_modelmeta().custom_metadata_map
        self.names: Dict[int, str] = ast.literal_eval(metadata["names"])

        used = threads or "default (one per core)"
        print(f"ONNX Runtime: {os.path.basename(path)}, input {self._input_w}x{self._input_h}, "
              f"{len(self.names)} classes, threads: {used}")

    def predict(self, img: np.ndarray, conf_threshold: float) -> List[Box]:
        """Detections in one BGR image, highest confidence first."""
        blob, scale, pad_x, pad_y = self._letterbox(img)
        output = self._session.run(None, {self._input_name: blob})[0][0]
        boxes = self._decode(output, conf_threshold)

        # Input pixels -> image pixels, clipped to the image.
        h, w = img.shape[:2]
        return [
            Box(b.class_id, b.confidence, (
                float(np.clip((b.xyxy[0] - pad_x) / scale, 0, w)),
                float(np.clip((b.xyxy[1] - pad_y) / scale, 0, h)),
                float(np.clip((b.xyxy[2] - pad_x) / scale, 0, w)),
                float(np.clip((b.xyxy[3] - pad_y) / scale, 0, h)),
            ))
            for b in boxes
        ]

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = min(self._input_h / h, self._input_w / w)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (self._input_w - new_w) / 2, (self._input_h - new_h) / 2
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)

        canvas = np.full((self._input_h, self._input_w, 3), PAD_VALUE, np.uint8)
        canvas[top:top + new_h, left:left + new_w] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # BGR HWC uint8 -> RGB CHW float [0, 1], with a batch axis.
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), scale, left, top

    def _decode(self, output: np.ndarray, conf_threshold: float) -> List[Box]:
        if output.shape[-1] == 6:
            # End-to-end export: already [x1, y1, x2, y2, conf, class] after NMS.
            rows = output[output[:, 4] > conf_threshold]
            return [Box(int(r[5]), float(r[4]), tuple(map(float, r[:4]))) for r in rows]

        predictions = output.T                      # [anchors, 4 + classes]
        scores = predictions[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        keep = confidences > conf_threshold
        if not keep.any():
            return []
        cx, cy, bw, bh = predictions[keep, :4].T
        class_ids, confidences = class_ids[keep], confidences[keep]

        # Per-class NMS, as ultralytics does by default (agnostic=False).
        rects = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
        kept = cv2.dnn.NMSBoxesBatched(rects.tolist(), confidences.tolist(), class_ids.tolist(),
                                       conf_threshold, self._iou_threshold)
        kept = sorted(np.array(kept).flatten(), key=lambda i: -confidences[i])[:MAX_DETECTIONS]
        return [
            Box(int(class_ids[i]), float(confidences[i]),
                (float(cx[i] - bw[i] / 2), float(cy[i] - bh[i] / 2),
                 float(cx[i] + bw[i] / 2), float(cy[i] + bh[i] / 2)))
            for i in kept
        ]
