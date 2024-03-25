import os
from pathlib import Path
import json
import numpy as np
import cv2
import pprint
import argparse


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
            point = np.array(shape["points"])
            x1,y1 = np.min(point,0)
            x2,y2 = np.max(point,0)
            type1 = shape["label"].lower()
            type1 = "_".join(type1.split())
            cls = category.index(type1)
            box = [cls, x1,y1,x2,y2]
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
        stat = (correct_bboxes, np.empty(0), np.empty(0), np.empty(0))
        
        # im = cv2.imread(str(img_path))
        # for b in preds:
        #     x1, y1, x2, y2, score, cls = b
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # for b in gt:
        #     cls,x1,y1, x2, y2 = b
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 2)
        # im = cv2.resize(im,(640,640))
        # cv2.imshow('im',im)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if len(preds)==0:
            if len(gt):
                stat = (correct_bboxes, np.empty(0), np.empty(0), gt[:, 0])
                stats.append(stat)
            continue
        if len(gt):
            correct_bboxes, _ = process_batch(preds, gt)
            stat = (correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 0])
            stats.append(stat)

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        metrics.process(*stats)
    result = metrics.results_dict
    pprint.pprint(result)


if __name__ == "__main__":
    main()
