"""Compose the minimal UI figure: model picker/comparison table + live metrics.

Takes the full 1440x1080 evaluator UI screenshot and crops the two information-
dense panels side by side so the text stays readable when printed.
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "reports" / "figures" / "ui"


def main():
    im = Image.open(UI / "ui_result.png").convert("RGB")
    # left: model picker + recorded comparison table
    left = im.crop((0, 410, 292, 690))
    # right: live metrics panel (top-1, continuity, sequence F1, macro F1, log loss, miss)
    right = im.crop((1110, 56, 1440, 430))
    gap = 16
    width = left.width + gap + right.width
    height = max(left.height, right.height) + 2
    canvas = Image.new("RGB", (width + 2, height + 2), "#ffffff")
    canvas.paste(left, (1, 1))
    canvas.paste(right, (1 + left.width + gap, 1))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline="#e5e5e5")
    out = UI / "ui_report.png"
    canvas.save(out)
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
