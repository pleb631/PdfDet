import warnings

warnings.filterwarnings("ignore")
import tempfile
import os
import cv2
import numpy as np
from pathlib import Path
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer as Trainer1
from ppdet.core.workspace import create


from pdfdet.models.baseModel import base_module
from pdfdet.utils import safe_download


class Trainer(Trainer1):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def predict(self, images):

        self.dataset.set_images(images)
        loader = create("TestReader")(self.dataset, 0)

        # Run Infer
        self.model.eval()

        data = next(loader)
        outs = self.model(data)

        for key, value in outs.items():
            if hasattr(value, "numpy"):
                outs[key] = value.numpy()

        return outs


def attempt_download(weight):
    name = Path(weight).stem
    if name == "picodet_lcnet_x1_0_fgd_layout_cdla":
        url = "https://github.com/pleb631/PdfDet/releases/download/v0.0.1/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams"
    elif name == "picodet_lcnet_x1_0_fgd_layout_pub":
        url = "https://github.com/pleb631/PdfDet/releases/download/v0.0.1/picodet_lcnet_x1_0_fgd_layout_pub.pdparams"
    else:
        raise ValueError()
    safe_download(file=weight, url=url)


class paddle_cdla(base_module):
    def __init__(self, *args, **kwargs) -> None:
        config = (
            os.path.dirname(__file__)
            + r"/configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout.yml"
        )
        weight = os.path.join(
            os.path.dirname(__file__),
            "weights/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams",
        )

        self.labels = {
            0: "Text",
            1: "Title",
            2: "Figure",
            3: "Figure caption",
            4: "Table",
            5: "Table caption",
            6: "Header",
            7: "Footer",
            8: "Reference",
            9: "Equation",
        }
        self.init(config, weight)

    def init(self, config, weight):
        cfg = load_config(config)
        trainer = Trainer(cfg, mode="test")
        attempt_download(weight)
        trainer.load_weights(weight)
        self.trainer = trainer

    def predict(self, path=None, image=None, *args, **kwargs):
        assert path is not None or image is not None
        if path is None:
            return self.predict_from_path(image)

        image = self.imread(path)
        path = [path]
        pred = self.trainer.predict(path)
        result = []
        pred = pred["bbox"]
        mask = pred[:, 1] > 0.25
        pred = pred[mask]
        for item in pred:
            cls, score, x1, y1, x2, y2 = item
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = self.labels[int(cls)]
            b = {"type": cls, "box": [x1, y1, x2, y2], "score": float(score)}
            result.append(b)

        return (result, image)

    def predict_from_path(self, image):
        with tempfile.NamedTemporaryFile(
            mode="wb+",
            suffix=f"{id(__file__)}.png",
            delete=False,
        ) as f:
            img_encode = cv2.imencode(".png", image)[1]

            data_encode = np.array(img_encode)
            str_encode = data_encode.tostring()
            f.write(str_encode)
            f.flush()
            path = f.name
            out = self.predict(path)
        os.remove(path)
        return out


class paddle_pub(paddle_cdla):
    def __init__(self, *args, **kwargs) -> None:
        config = (
            os.path.dirname(__file__)
            + r"/configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout1.yml"
        )
        weight = os.path.join(
            os.path.dirname(__file__),
            "weights/picodet_lcnet_x1_0_fgd_layout_pub.pdparams",
        )

        self.labels = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

        self.init(config, weight)
