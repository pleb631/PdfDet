import argparse
import tqdm
from pathlib import Path
import json
import sys
import os
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)

from pdfdet import uni_model


def save_json(json_path, info, indent=4, mode="w", with_return_char=False):
    json_str = json.dumps(info, indent=indent, ensure_ascii=True)
    if with_return_char:
        json_str += "\n"

    with open(json_path, mode, encoding="UTF-8") as json_file:
        json_file.write(json_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Process Images for Layout Analysis"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="paddle_cdla",
        choices=["paddle_pub", "paddle_cdla" "cnstd_yolov7", "yolov8m_cdla","yolov8n_cdla","yolov8l_doc","yolov8n_doc","yolov8s_doc"],
        help="Choose the detection model")
    parser.add_argument(
        "--src",
        type=str,
        help="Source folder containing images.",
        default=r"E:\project\code\labelme\Layout-Analysis-main\Layout-Analysis-main\layout_modify",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Destination folder to save results.",
        default=r"./result",
    )
    ret = parser.parse_args()
    return ret


def main():
    """
    Main function for batch processing images for layout analysis.
    """
    args = parse_args()
    model = uni_model(name=args.model)
    for item in tqdm.tqdm(list(Path(args.src).rglob("*.png"))):
        doc = model(path=str(item))
        content = doc.to_json()
        save_path = Path(args.save) / (item.relative_to(args.src)).with_suffix(".json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(save_path, content["boxes"])


if __name__ == "__main__":
    main()
