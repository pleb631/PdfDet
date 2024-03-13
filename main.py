import cv2
import os
import argparse

root = os.path.dirname(__file__)


from pdfdet import uni_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="paddle_cdla_model",
        choices=[
            "paddle_pub_model",
            "paddle_cdla_model",
            "cnstd_model",
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
    layers = sorted(doc.layers, key=lambda x: int(x))
    for i in layers:
        layer = getattr(doc, i)
        im = layer.imshow()
        cv2.imshow("im", im)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    content = doc.to_json()


if __name__ == "__main__":
    main()
