import warnings

warnings.filterwarnings("ignore")
import os
import sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 1)))
sys.path.insert(0, parent_path)

from .paddle_cdla import paddle_cdla



class paddle_pub(paddle_cdla):
    def __init__(self, *args, **kwargs) -> None:
        config = (
            os.path.dirname(__file__)
            + r"/configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout1.yml"
        )
        weight = os.path.join(os.path.dirname(__file__),"weights/picodet_lcnet_x1_0_fgd_layout_pub.pdparams")
        
        self.labels = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

        self.init(config, weight)
