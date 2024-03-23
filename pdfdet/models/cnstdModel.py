import numpy as np
import warnings
from cnstd import LayoutAnalyzer


warnings.filterwarnings("ignore")


from .baseModel import base_module


class cnstd_yolov7(base_module):
    def __init__(self, *args) -> None:
        self.analyzer = LayoutAnalyzer(
            "layout",
            device="cpu",
        )

    def predict(self, image=None, path=None, *args, **kwargs):

        assert path is not None or image is not None
        if path is not None:
            image = self.imread(path)
        elif not isinstance(image, np.ndarray):
            raise NotImplementedError

        pred = self.analyzer.analyze(image,conf_threshold=0.25)
        result = []
        for box in pred:
            b = {
                "type": box["type"],
                "box": box["box"][[0, 2]].reshape(-1).tolist(),
                "score": float(box["score"]),
            }
            result.append(b)

        return (result, image)
