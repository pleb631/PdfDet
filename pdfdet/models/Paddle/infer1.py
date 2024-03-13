import os
import sys
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer

# ignore warning log
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    paddle_path = r"D:\project\labelme\PdfDet\pdfdet\models\Paddle"
    config = r"/configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout1.yml"
    weight = r"/weights/picodet_lcnet_x2_5_layout/model.pdparams"
    images = [
        r"D:\project\labelme\Layout-Analysis-main\Layout-Analysis-main\layout_modify\0423.pdf\0.png"
    ]
    cfg = load_config(paddle_path + config)
    trainer = Trainer(cfg, mode="test")
    trainer.load_weights(paddle_path + weight)
    out = trainer.predict(images)
    import cv2

    im = cv2.imread(images[0])
    en = {
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
    out = out[0]["bbox"]
    mask = out[:, 1] > 0.5
    out = out[mask]
    for item in out:
        cls, score, x1, y1, x2, y2 = item
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
        cv2.putText(
            im,
            en[int(cls)],
            (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            [255, 0, 0],
            4,
        )
    im = cv2.resize(im, (512, 512))
    cv2.imshow("im", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ...
