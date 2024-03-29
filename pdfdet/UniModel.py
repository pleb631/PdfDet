import importlib

name2func = {
    "cnstd_yolov7": "pdfdet.models.cnstdModel",
    "paddle_pub": "pdfdet.models.Paddle",
    "paddle_cdla": "pdfdet.models.Paddle",
    "yolov8":"pdfdet.models.yolov8",
}


def uni_model(name=None, *args, **kwargs):
    name = name.lower()
    if 'yolov8' in name:
        kwargs["model_type"]=name
        name = 'yolov8'
        
    module = importlib.import_module(name2func[name])
    model = getattr(module, name)(*args, **kwargs)

    return model
