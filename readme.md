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

| 模型              | 来源   | 关联数据集        |
| ----------------- | ------ | ----------------- |
| paddle_pub_model  | paddle | PubLayNet（英文） |
| paddle_cdla_model | paddle | CDLA（中文）      |
| cnstd_model       | cnstd  | CDLA              |
