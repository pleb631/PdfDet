import os
import fitz

def pdf2png(pdf_path, save_dir):
    pdfDoc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        pix = page.get_pixmap(alpha=False)   
        if not save_dir.lstrip(pdf_name):
            save_path = os.path.join(save_dir, pdf_name,f'{pg}.png')
        else:
            save_path = os.path.join(save_dir, f'{pg}.png')
        os.makedirs(os.path.dirname(save_path),exist_ok=True)# 默认是720*x尺寸
        pix._writeIMG(save_path,'png',100)


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)