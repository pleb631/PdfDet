import importlib

name2func = {
    "cnstd_model": "pdfdet.models.cnstdModel",
    "paddle_pub_model": "pdfdet.models.Paddle",
    "paddle_cdla_model": "pdfdet.models.Paddle",
}


def uni_model(name=None, *args, **kwargs):

    module = importlib.import_module(name2func[name])
    model = getattr(module, name)(*args, **kwargs)

    return model
