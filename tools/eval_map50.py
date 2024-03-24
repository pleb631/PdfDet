import os
from pathlib import Path
import json
import numpy as np
import cv2
import pprint
import argparse
import sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)

from pdfdet.utils.DetMetrics import DetMetrics, process_batch


def read_json(json_path, mode="all"):
    json_data = []
    with open(json_path, "r", encoding="UTF-8") as json_file:
        if mode == "all":
            # Convert the read content to a Python dictionary
            json_data = json.loads(json_file.read())
        elif mode == "line":
            for line in json_file:
                json_line = json.loads(line)
                json_data.append(json_line)
    return json_data


def get_label_from_file(path, category):
    """
    Extract labels from a JSON file.

    Args:
        path (str): Path to the JSON file.
        category (list): List of category labels.

    Returns:
        np.ndarray: Extracted bounding boxes.
    """
    content = read_json(path)
    boxes = []
    labels = []
    if "shapes" in content and len(content["shapes"]) > 0:
        for shape in content["shapes"]:
            point = shape["points"]
            type1 = shape["label"].lower()
            type1 = "_".join(type1.split())
            cls = category.index(type1)
            box = [cls, *point[0], *point[2]]
            boxes.append(box)
    return np.array(boxes)


def get_pred_from_file(path, category):
    """
    Extract predictions from a JSON file.

    Args:
        path (str): Path to the JSON file.
        category (list): List of category labels.

    Returns:
        np.ndarray: Extracted predictions.
    """
    content = read_json(path)
    boxes = []
    for item in content:
        type1 = item["label"].lower()
        type1 = "_".join(type1.split())
        if type1 not in category:
            tp = {
                "footnote": "footer",
                "formula": "equation",
                "page-footer": "footer",
                "page-header": "header",
                "picture": "figure",
                "section-header": "header",
                "caption": "figure_caption",
                "list-item": "text",
            }
            type1 = tp[type1]
        cls = category.index(type1)
        boxes.append([*item["box"], item["score"], cls])
    return np.array(boxes)


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Predictions")
    parser.add_argument("gt_root", type=str, help="Path to the ground truth directory")
    parser.add_argument(
        "pred_root", type=str, help="Path to the predicted JSON directory"
    )
    ret = parser.parse_args()
    return ret


def main():
    """
    Main function for evaluating predictions.
    """
    opt = parse_args()
    root = opt.gt_root
    pred_dir = opt.pred_root
    category = [
        "text",
        "title",
        "figure",
        "figure_caption",
        "table",
        "table_caption",
        "header",
        "footer",
        "reference",
        "equation",
    ]

    root = Path(root)
    pred_dir = Path(pred_dir)

    metrics = DetMetrics(names=category, plot=False)

    items = root.rglob("*.png")
    stats = []
    for img_path in items:
        json_path = img_path.with_suffix(".json")
        pred_path = pred_dir / json_path.relative_to(root)
        gt = get_label_from_file(json_path, category)
        preds = get_pred_from_file(pred_path, category)
        correct_bboxes = np.zeros((len(preds), 10))
        matches = np.empty((0, 2))
        stat = (correct_bboxes, np.empty(0), np.empty(0), np.empty(0))

        if len(gt) > 0 and len(preds) == 0:
            stat = (correct_bboxes, np.empty(0), np.empty(0), gt[:, 0])
        elif len(preds) > 0 and len(gt) > 0:
            correct_bboxes, matches = process_batch(preds, gt)
            stat = (correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 0])
        elif len(preds) > 0 and len(gt) == 0:
            stat = (correct_bboxes, preds[:, 4], preds[:, 5], np.empty(0))

        stats.append(stat)

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        metrics.process(*stats)
    result = metrics.results_dict
    pprint.pprint(result)


if __name__ == "__main__":
    main()
