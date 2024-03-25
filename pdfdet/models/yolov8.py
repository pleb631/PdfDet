# https://huggingface.co/egis-group/LayoutDetection

import numpy as np
import os
from pathlib import Path
try:
    import ultralytics
except:
    os.system('pip install ultralytics')

from ultralytics.models.yolo.detect import DetectionPredictor



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
    if name in ["yolov8l_doc","yolov8s_doc","yolov8n_doc"] :
        url = f"https://github.com/pleb631/PdfDet/releases/download/v0.0.1/{name}.onnx"
    elif name in ["yolov8m_cdla","yolov8n_cdla"]:
        url = f"https://github.com/pleb631/PdfDet/releases/download/v0.0.1/{name}.pt"
    else:
        raise ValueError()
    safe_download(file=weight, url=url)


class yolov8(base_module):
    def __init__(self, model_type="yolov8l_doc",*args,**kwargs) -> None:
        if model_type in ["yolov8l_doc","yolov8s_doc","yolov8n_doc"] :
            weight = os.path.join(os.path.dirname(__file__), "weights",model_type+'.onnx')
        else:
            weight = os.path.join(os.path.dirname(__file__), "weights",model_type+'.pt')
        attempt_download(weight)
        args = dict(
            model=weight,
        )
        self.analyzer = DetectionPredictor1(overrides=args)
        if model_type.split('_')[-1]=='doc':
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
        else:
            self.labels = {0: 'Header', 1: 'Text', 2: 'Reference', 3: 'Figure caption', 4: 'Figure', 5: 'Table caption', 6: 'Table', 7: 'Title', 8: 'Footer', 9: 'Equation'}

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
