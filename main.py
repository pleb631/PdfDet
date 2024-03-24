import cv2
import os
import argparse

root = os.path.dirname(__file__)


from pdfdet import uni_model
from pdfdet.utils.instance import Layer, Document


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="paddle_cdla",
        choices=[
            "paddle_pub",
            "paddle_cdla",
            "cnstd_yolov7",
            "yolov8"
        ],
    )
    parser.add_argument(
        "--path",
        type=str,
        help="pdf地址或单个图片地址,不支持文件夹地址",
        default=root + r"/source/2309.12585.pdf",
    )
    ret = parser.parse_args()
    return ret


def main():
    args = parse_args()
    model = uni_model(name=args.model)
    doc = model(path=args.path)
    if isinstance(doc, Layer):
        im = doc.imshow()
        im = cv2.resize(im,(640,640))
        cv2.imshow("im", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif isinstance(doc, Document):
        layers = sorted(doc.layers, key=lambda x: int(x))
        for i in layers:
            layer = getattr(doc, i)
            im = layer.imshow()
            im = cv2.resize(im,(640,640))
            cv2.imshow("im", im)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    content = doc.to_json()


if __name__ == "__main__":
    main()
