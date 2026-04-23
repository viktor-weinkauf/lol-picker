"""Generate app icon (icon.ico) and tray icon (tray.png) for LoLPicker.
Geometry mirrors HalfHexFinal.jsx — a hexagon bisected vertically, with a
thin colored accent seam. Re-run this script to regenerate after tweaks."""
import os
from PIL import Image, ImageDraw

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# Colors (HalfHexFinal "dark" theme + "#3d7aed" accent)
BG_FROM = (26, 29, 36)
BG_TO = (12, 14, 19)
FG = (236, 234, 228)
ACCENT = (61, 122, 237)
RIM = (255, 255, 255, 15)


def _vertical_gradient(size):
    img = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        c = tuple(int(BG_FROM[i] + (BG_TO[i] - BG_FROM[i]) * t) for i in range(3))
        draw.line([(0, y), (size, y)], fill=c + (255,))
    return img


def make_app_icon(size=512):
    """Dark squircle tile with the bisected hex mark + blue accent seam."""
    squircle_r = round(112 * size / 512)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=squircle_r, fill=255
    )
    img.paste(_vertical_gradient(size), (0, 0), mask)

    draw = ImageDraw.Draw(img)
    scale = size / 512
    cx, cy = size // 2, size // 2
    hx = round(156 * scale)
    hy = round(176 * scale)
    hy2 = round(88 * scale)

    pts = [
        (cx, cy - hy),
        (cx + hx, cy - hy2),
        (cx + hx, cy + hy2),
        (cx, cy + hy),
        (cx - hx, cy + hy2),
        (cx - hx, cy - hy2),
    ]
    stroke_w = max(1, round(14 * scale))

    draw.polygon(pts, outline=FG + (255,), width=stroke_w)
    left_half = [pts[0], pts[5], pts[4], pts[3]]
    draw.polygon(left_half, fill=FG + (255,))

    seam_w = max(2, round(10 * scale))
    seam_inset = round(22 * scale)
    draw.rounded_rectangle(
        [cx - seam_w // 2, cy - hy + seam_inset,
         cx + seam_w // 2, cy + hy - seam_inset],
        radius=seam_w // 2,
        fill=ACCENT + (255,),
    )

    rim_overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(rim_overlay).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=squircle_r, outline=RIM, width=1
    )
    img = Image.alpha_composite(img, rim_overlay)
    return img


def make_tray_icon(size=64):
    """Monochrome hex silhouette on transparent background for the tray."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    scale = size / 32
    pts = [
        (round(16 * scale), round(3 * scale)),
        (round(27 * scale), round(9 * scale)),
        (round(27 * scale), round(23 * scale)),
        (round(16 * scale), round(29 * scale)),
        (round(5 * scale), round(23 * scale)),
        (round(5 * scale), round(9 * scale)),
    ]
    stroke_w = max(1, round(2 * scale))
    white = (255, 255, 255, 255)
    draw.polygon(pts, outline=white, width=stroke_w)
    left_half = [pts[0], pts[5], pts[4], pts[3]]
    draw.polygon(left_half, fill=white)
    return img


if __name__ == "__main__":
    sizes = [256, 128, 64, 48, 32, 16]
    base = make_app_icon(256)
    base.save(
        os.path.join(ASSETS_DIR, "icon.ico"),
        format="ICO",
        sizes=[(s, s) for s in sizes],
    )
    print(f"Wrote icon.ico ({', '.join(f'{s}x{s}' for s in sizes)})")

    make_tray_icon(64).save(os.path.join(ASSETS_DIR, "tray.png"))
    print("Wrote tray.png (64x64)")

    # Browser favicon — same as app icon, dropped into static/ so Flask serves it
    static_dir = os.path.join(os.path.dirname(ASSETS_DIR), "static")
    base.save(
        os.path.join(static_dir, "favicon.ico"),
        format="ICO",
        sizes=[(32, 32), (16, 16)],
    )
    print("Wrote static/favicon.ico (32x32, 16x16)")
