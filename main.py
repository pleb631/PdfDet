import cv2
import os
import argparse

root = os.path.dirname(__file__)

from pdfdet import uni_model
from pdfdet.utils import Layer, Document


def parse_args():
    parser = argparse.ArgumentParser(description="PDF Layout Detection Toolbox")

    parser.add_argument(
        "--model",
        type=str,
        default="paddle_cdla",
        choices=["paddle_pub", "paddle_cdla" "cnstd_yolov7", "yolov8m_cdla","yolov8n_cdla","yolov8l_doc","yolov8n_doc","yolov8s_doc"],
        help="Choose the detection model",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="PDF file path or path to a single image. Folder paths are not supported.",
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
        im = cv2.resize(im, (640, 640))
        cv2.imshow("im", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif isinstance(doc, Document):
        layers = sorted(doc.layers, key=lambda x: int(x))
        for i in layers:
            layer = getattr(doc, i)
            im = layer.imshow()
            im = cv2.resize(im, (640, 640))
            cv2.imshow("im", im)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    content = doc.to_json()


if __name__ == "__main__":
    main()
