# pdfdet: pdf文件版面检测工具箱

![效果1](./source/1.jpg)

## 环境要求

python>=3.10

```shell
python -m pip install -r requirements.txt
```

## 使用

```shell
python main.py
```

## 模型

| 模型         | 来源   | 关联数据集        | map50:95 | p      | r      |
| ------------ | ------ | ----------------- | -------- | ------ | ------ |
| paddle_pub   | paddle | PubLayNet（英文） | 0.0486   | 0.1133 | 0.1000 |
| paddle_cdla  | paddle | CDLA（中文）      | 0.5717   | 0.5853 | 0.6248 |
| cnstd_yolov7 | cnstd  | CDLA              | 0.5034   | 0.6278 | 0.5651 |
| yolov8       | [huggingface](https://huggingface.co/egis-group/LayoutDetection)      | -                 | 0.2483   | 0.3785 | 0.3333 |

[测试集](https://github.com/Ontheroad123/Layout-Analysis/tree/main/layout_modify)

[评测代码来源](https://github.com/ultralytics/ultralytics/blob/2d513a9e4bf51e961a4199067383d2052f483874/ultralytics/utils/metrics.py#L620)

**注**:不同数据集的标签形式不统一，标注策略也有差别，对比效果以可视化为主
