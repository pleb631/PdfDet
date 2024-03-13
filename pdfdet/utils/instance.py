import numpy as np
import cv2
from .utils import compute_color_for_labels


class Bbox:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'ltrb', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray| List): The bounding boxes stored in a 1D numpy array.
        format (str): The format of the bounding boxes ('ltrb', 'xywh', or 'ltwh').
    """

    _formats = ["ltrb", "xywh", "ltwh"]

    def __init__(self, box, format="ltrb", label=None, score=None) -> None:
        assert (
            format in Bbox._formats
        ), f"Invalid bounding box format: {format}, format must be one of {Bbox._formats}"
        if isinstance(box, list):
            box = np.array(box)
        assert box.ndim == 1
        assert box.shape[0] == 4

        self.__label = label
        self.__bbox = Bbox.convert(box, format, "ltrb")
        self.__score = score

    @property
    def ltrb(self) -> np.ndarray:
        return self.__bbox

    @property
    def label(self) -> np.ndarray:
        return self.__label

    @property
    def label(self) -> np.ndarray:
        return self.__score

    @staticmethod
    def convert(box, srcf, dstf):
        assert (
            srcf in Bbox._formats
        ), f"Invalid bounding box format: {srcf}, format must be one of {box._formats}"
        assert (
            dstf in Bbox._formats
        ), f"Invalid bounding box format: {dstf}, format must be one of {box._formats}"
        if srcf == dstf:
            return box
        fun_str = f"Bbox.{srcf}2{dstf}"
        try:
            func = eval(fun_str)
        except:
            raise NotImplementedError(
                f"Conversion from {srcf} to {dstf} is not implemented"
            )
        return func(box)

    @staticmethod
    def ltrb2xywh(x):
        y = np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    @staticmethod
    def xywh2ltrb(x):
        y = np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = y[..., 0] + x[..., 2]  # bottom right x
        y[..., 3] = y[..., 1] + x[..., 3]  # bottom right y
        return y

    @staticmethod
    def ltrb2ltwh(x):
        y = np.copy(x)
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    @staticmethod
    def ltwh2ltrb(x):
        y = np.copy(x)
        y[..., 2] = x[..., 2] + x[..., 0]  # width
        y[..., 3] = x[..., 3] + x[..., 1]  # height
        return y

    def __repr__(self) -> str:
        return f"ltrb: {self.__bbox.tolist()}  label: {self.__label}"

    def to_json(self):
        return {
            "box": self.__bbox.tolist(),
            "label": self.__label,
            "score": self.__score,
        }

    @classmethod
    def from_xyxy(
        cls,
        x1,
        y1,
        x2,
        y2,
        label,
        score,
        page_width=None,
        page_height=None,
        is_Pixel_distance=False,
    ):

        assert x2 >= x1, "Requires x2 >= x1"
        assert y2 >= y1, "Requires y2 >= y1"

        if is_Pixel_distance:
            if not page_height is None and not page_width is None:
                x1, x2 = x1 * page_width, x2 * page_width
                y1, y2 = y1 * page_height, y2 * page_height
            else:
                raise ValueError("Requires page_width and page_height")

        box = [int(x1), int(y1), int(x2), int(y2)]
        return cls(box, label=label, score=score)

    def plt(self, image):

        if not image is None:
            x1, y1, x2, y2 = self.__bbox.tolist()
            tl = (
                round(0.001 * (image.shape[0] + image.shape[1]) / 2) + 1
            )  # line/font thickness
            color = compute_color_for_labels(int(self.__label.encode().hex(), 16) % 256)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, tl)
            cv2.putText(
                image,
                self.__label,
                (int(x1) + 5, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                tl,
            )


class Layer:
    def __init__(
        self,
        boxes=None,
        image=None,
    ):

        if not boxes is None:
            for b in boxes:
                b.layer = self
        self.__boxes = boxes if boxes is not None else []
        self.__image = image if image is not None else []

    def to_json(self):
        return {"image": self.__image, "boxes": [box.to_json() for box in self.__boxes]}

    def imshow(
        self,
    ):
        im = self.__image.copy()
        for box in self.__boxes:
            box.plt(im)

        return im

    @property
    def boxes(self):
        return self.__boxes

    @property
    def image(self):
        return self.__image


class Document:
    def __init__(
        self,
        layers=None,
    ) -> None:
        self.__layers = layers if layers is not None else []

    def annotate_layer(self, name: str, entities: Layer | Bbox, image=None) -> None:
        self.validate_layer_name_availability(name=name)

        if isinstance(entities, Bbox):
            layer = Layer(entities, image)
        else:
            layer = entities

        layer.doc = self
        layer.name = name
        setattr(self, name, layer)
        self.__layers.append(name)

    def validate_layer_name_availability(self, name: str) -> None:
        if name in self.__layers:
            raise AssertionError(
                f'{name} already exists. Try `doc.remove_layer("{name}")` first.'
            )
        if name in dir(self):
            raise AssertionError(f"{name} clashes with Document class properties.")

    def remove_layer(self, name: str):
        if name not in self.__layers:
            pass
        else:
            getattr(self, name).doc = None
            getattr(self, name).name = None
            delattr(self, name)
            self._layers.remove(name)

    def to_json(self, layers=None):

        # 1) instantiate basic Document dict
        doc_dict = {
            "layers": {},
        }

        # 2) serialize each layer to JSON
        layers = self.__layers if layers is None else layers
        for layer in layers:
            doc_dict["layers"][layer] = getattr(self, layer).to_json()

        return doc_dict

    @property
    def layers(self):
        return self.__layers
