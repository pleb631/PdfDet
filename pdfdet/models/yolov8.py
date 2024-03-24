# https://huggingface.co/egis-group/LayoutDetection

import numpy as np
import warnings
from ultralytics.models.yolo.detect import DetectionPredictor
from pathlib import Path

warnings.filterwarnings("ignore")
import os

from .baseModel import base_module
from pdfdet.utils.utils import safe_download


class DetectionPredictor1(DetectionPredictor):
    def predict_cli(self, source=None, model=None):
        gen = self.stream_inference(source, model)
        for (
            _
        ) in (
            gen
        ):  # noqa, running CLI inference without accumulating any outputs (do not modify)
            return _.boxes.data


def attempt_download(weight):
    name = Path(weight).stem
    if name == "yolov8l":
        url = "https://github.com/pleb631/PdfDet/releases/download/v0.0.1/yolov8l.onnx"
    elif name == "yolov8s":
        url = "https://github.com/pleb631/PdfDet/releases/download/v0.0.1/yolov8s.onnx"
    elif name == "yolov8n":
        url = "https://github.com/pleb631/PdfDet/releases/download/v0.0.1/yolov8n.onnx"
    else:
        raise ValueError()
    safe_download(file=weight, url=url)


class yolov8(base_module):
    def __init__(self, *args) -> None:
        weight = os.path.join(os.path.dirname(__file__), "weights/yolov8l.onnx")
        attempt_download(weight)
        args = dict(
            model=weight,
        )
        self.analyzer = DetectionPredictor1(overrides=args)

        self.labels = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
        }

    def predict(self, image=None, path=None, *args, **kwargs):

        assert path is not None or image is not None
        if path is not None:
            image = self.imread(path)
        elif not isinstance(image, np.ndarray):
            raise NotImplementedError

        pred = self.analyzer.predict_cli(source=image)

        result = []
        for box in pred:
            box = box.reshape(-1).tolist()
            b = {
                "type": self.labels[int(box[-1])],
                "box": box[:4],
                "score": box[-2],
            }
            result.append(b)

        return (result, image)
