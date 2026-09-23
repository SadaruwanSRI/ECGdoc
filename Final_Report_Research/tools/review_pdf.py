"""Render every PDF page and contact sheets for final visual review."""
import argparse
from pathlib import Path
import fitz
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
doc = fitz.open(args.pdf)
landscape = doc[0].rect.width > doc[0].rect.height
thumb_w = 700 if landscape else 430
thumb_h = 405 if landscape else 625
cols, rows = (2, 3) if landscape else (3, 2)
for index, page in enumerate(doc):
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(args.output / f"page-{index+1:03}.png")
for start in range(0, len(doc), cols * rows):
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for local, index in enumerate(range(start, min(start + cols * rows, len(doc)))):
        pic = Image.open(args.output / f"page-{index+1:03}.png")
        pic.thumbnail((thumb_w-12, thumb_h-25))
        x, y = (local % cols) * thumb_w, (local // cols) * thumb_h
        sheet.paste(pic, (x+6, y+22))
        draw.text((x+8, y+4), f"Page {index+1}", fill="black")
    sheet.save(args.output / f"sheet-{start//(cols*rows)+1:02}.png")
print(f"Rendered {len(doc)} pages to {args.output}")
