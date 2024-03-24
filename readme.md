# pdfdet: PDF Layout Detection Toolbox

![效果1](./source/1.jpg)

## Requirements

python>=3.10

```shell
python -m pip install -r requirements.txt
```

## Usage

### Simple Operations

```bash
python main.py
```

### Batch Operations

```bash
# batch predict
python tools/batch_process.py --model "model_name" --src "image_root" --save "res_root"
# evaluate dataset
python tools/eval_map50.py "gt_root" "res_root"
# generate visualize result
python tools/visualize.py "image_path" "res_path"
```

## Models

| **Model** | **Source** | **Associated Dataset** | **map50:95** | **p** | **r** |
|---|---|---|---|---|---|
| **paddle_pub** | [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/layout/README_ch.md) | PubLayNet(English) | 0.0486 | 0.1133 | 0.1000 |
| **paddle_cdla** | [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/layout/README_ch.md) | CDLA(Chinese) | 0.5717 | 0.5853 | 0.6248 |
| **cnstd_yolov7** | [CNSTD](https://github.com/breezedeus/cnstd) | CDLA | 0.5034 | 0.6278 | 0.5651 |
| **yolov8** | [huggingface](https://huggingface.co/egis-group/LayoutDetection) | DocLayNet(English, German,French, Japanese) | 0.2483 | 0.3785 | 0.3333 |


[Test Dataset](https://github.com/Ontheroad123/Layout-Analysis/tree/main/layout_modify)

[Evaluation Code Source](https://github.com/ultralytics/ultralytics/blob/2d513a9e4bf51e961a4199067383d2052f483874/ultralytics/utils/metrics.py#L620)

**Note**: Labels and annotation strategies vary across different datasets. Visual comparison should be the primary method for evaluating effectiveness.
