import os
from pathlib import Path
import json
import numpy as np
import cv2
import pprint

from pdfdet.utils.DetMetrics import DetMetrics,process_batch

def read_json(json_path, mode='all'):
    '''读取json文件

    Args:
        json_path: str, json文件路径
        mode: str, 'all'模式代表一次性读取json文件全部内容,只存在一个字典；'line'模式代表按行读取json文件内容,每行为一个字典

    Returns:
        json_data: list, json文件内容
    '''
    json_data = []
    with open(json_path, 'r',encoding="UTF-8") as json_file:
        if mode == 'all':
            # 把读取内容转换为python字典
            json_data = json.loads(json_file.read())
        elif mode == 'line':
            for line in json_file:
                json_line = json.loads(line)
                json_data.append(json_line)

    return json_data

def get_label_from_file(path,category):
    content = read_json(path)
    boxes = []
    labels = []
    if "shapes" in content and len(content["shapes"])>0:
        for shape in content["shapes"]:
            
            point = shape["points"]
            type1 = shape["label"].lower()
            type1= '_'.join(type1.split())
            cls = category.index(type1)
            box = [cls,*point[0], *point[2]]
            
            boxes.append(box)

    return np.array(boxes)


def get_pred_from_file(path,category):
    content = read_json(path)
    boxes = []
    for item in content:
        type1 = item["label"].lower()
        type1= '_'.join(type1.split())
        if type1 not in category:
            tp = {'footnote':"footer","formula":"equation",'page-footer':"footer",'page-header':"header",'picture':"figure",'section-header':"header",'caption':"figure_caption",'list-item':"text"}
            type1 = tp[type1]
        cls = category.index(type1)
        boxes.append([*item["box"],item['score'],cls])
        
    return np.array(boxes)

def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)

if __name__=='__main__':
    root = r'E:\project\code\labelme\Layout-Analysis-main\Layout-Analysis-main\layout_modify'
    pred_dir = r'E:\project\code\PdfDet\result\yolov8l'
    category = ["text","title","figure","figure_caption","table","table_caption","header","footer","reference","equation"]
    
    root = Path(root)
    pred_dir = Path(pred_dir)
    
    metrics = DetMetrics(names=category, plot=False)
    
    items = root.rglob('*.png')
    stats=[]
    for img_path in items:
        
        json_path = img_path.with_suffix('.json')
        pred_path = pred_dir/json_path.relative_to(root)
        gt = get_label_from_file(json_path,category)
        preds = get_pred_from_file(pred_path,category)
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