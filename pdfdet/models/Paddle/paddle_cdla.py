import warnings
import tempfile

warnings.filterwarnings("ignore")
import os
import cv2
import sys
import numpy as np

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 1)))
sys.path.insert(0, parent_path)


from pdfdet.models.baseModel import base_module
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer as Trainer1
from ppdet.core.workspace import create

class Trainer(Trainer1):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def predict(self,
                images):

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        # Run Infer 
        self.model.eval()
        results = []
        for step_id, data in enumerate(loader):
            # forward
            outs = self.model(data)

            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        return results

class paddle_cdla_model(base_module):
    def __init__(self, *args, **kwargs) -> None:
        config = (
            os.path.dirname(__file__)
            + r"/configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout.yml"
        )
        weight = (
            os.path.dirname(__file__)
            + r"/weights/picodet_lcnet_x1_0_fgd_layout_cdla/model.pdparams"
        )
        cfg = load_config(config)
        trainer = Trainer(cfg, mode="test")
        trainer.load_weights(weight)
        self.trainer = trainer
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

    def predict(self, path=None, image=None, *args, **kwargs):
        assert path is not None or image is not None
        if path is None:
            return self.predict_from_path(image)

        image = self.imread(path)
        path = [path]
        pred = self.trainer.predict(path)
        result = []
        pred = pred[0]["bbox"]
        mask = pred[:, 1] > 0.5
        pred = pred[mask]
        for item in pred:
            cls, score, x1, y1, x2, y2 = item
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = self.labels[int(cls)]
            b = {"type": cls, "box": [x1, y1, x2, y2], "score": score}
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
