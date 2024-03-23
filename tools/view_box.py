import cv2
import json
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def read_json(json_path, mode="all"):
    json_data = []
    with open(json_path, "r", encoding="UTF-8") as json_file:
        if mode == "all":
            # 把读取内容转换为python字典
            json_data = json.loads(json_file.read())
        elif mode == "line":
            for line in json_file:
                json_line = json.loads(line)
                json_data.append(json_line)

    return json_data

im = r'E:\project\code\labelme\Layout-Analysis-main\Layout-Analysis-main\layout_modify\0029.pdf\4.png'
pred = r'E:\project\code\PdfDet\result\0029.pdf\4.json'

im = imread(im)
anno = read_json(pred)
for item in anno:
    box = item["box"]
    box =list(map(int,box))
    category = item['label']
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
    cv2.putText(im,category,(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,2,[0,255,0],2)
im = cv2.resize(im,(640,640))
cv2.imshow("im",im)
cv2.waitKey(0)
cv2.destroyAllWindows()
