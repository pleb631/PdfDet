import warnings

warnings.filterwarnings("ignore")
import os
import sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 1)))
sys.path.insert(0, parent_path)

from .paddle_cdla import paddle_cdla_model
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer


class paddle_pub_model(paddle_cdla_model):
    def __init__(self, *args, **kwargs) -> None:
        config = (
            os.path.dirname(__file__)
            + r"/configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout1.yml"
        )
        weight = (
            os.path.dirname(__file__)
            + r"/weights/picodet_lcnet_x2_5_layout/model.pdparams"
        )
        cfg = load_config(config)
        trainer = Trainer(cfg, mode="test")
        trainer.load_weights(weight)
        self.trainer = trainer
        self.labels = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
