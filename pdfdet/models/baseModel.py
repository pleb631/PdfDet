from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import tqdm

from pdfdet.utils import Bbox, Layer, Document


class base_module(metaclass=ABCMeta):
    image_format = [".jpg", ".png", ".jpeg", ".bmp"]

    def imread(self, path, flags=cv2.IMREAD_COLOR):
        return cv2.imdecode(np.fromfile(path, np.uint8), flags)

    def __call__(self, path=None, image=None, *args, **kwargs):
        if image is not None:
            return self.single_predict(image=image, *args, **kwargs)
        elif path is not None:
            if Path(path).suffix.lower() == ".pdf":
                return self.pdf_predict(pdf_path=path, *args, **kwargs)
            elif Path(path).suffix.lower() in base_module.image_format:
                return self.single_predict(path=path, *args, **kwargs)
            else:
                raise ValueError("Unsupported file format")

    def single_predict(self, *args, **kwargs):
        preds, image = self.predict(*args, **kwargs)
        result = []
        for p in preds:
            label, box, score = (
                p.get("type", None),
                p.get("box", None),
                p.get("score", None),
            )
            result.append(Bbox.from_xyxy(*box, label, score))
        return Layer(result, image)

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def pdf_predict(self, pdf_path, *args, **kwargs):
        """
        Predict on a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Document: Annotated document with predictions.
        """
        from pdfdet.utils import pdf2png

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_content = Document()
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_png_dir = os.path.join(temp_dir, pdf_name)
            pdf2png(pdf_path, pdf_png_dir)
            for png_path in tqdm.tqdm(list(Path(pdf_png_dir).rglob("*.png")), ncols=50):
                layer_name = Path(png_path).stem
                pred = self.single_predict(path=str(png_path), *args, **kwargs)
                pdf_content.annotate_layer(name=layer_name, entities=pred)
        return pdf_content
