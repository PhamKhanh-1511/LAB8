"""Tạo 3 ảnh MINH HỌA (vẽ bằng matplotlib) cho 3 loài Iris vào thư mục images/.

Đây là hình vẽ cách điệu, không phải ảnh chụp thật. Muốn dùng ảnh thật chỉ cần chép đè
images/setosa.jpg, images/versicolor.jpg, images/virginica.jpg.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# (màu cánh đứng, màu cánh rủ, màu vệt giữa, độ to cánh, nền)
STYLE = {
    "setosa": ("#6a5acd", "#4b3fb5", "#f4e04d", 0.8, "#e8f1fb"),
    "versicolor": ("#8e44ad", "#6c3483", "#f1c40f", 1.0, "#eef7ec"),
    "virginica": ("#4a2c8a", "#2e1a5e", "#f5b041", 1.25, "#fbeff1"),
}


def draw(species: str, out_dir: str = "images"):
    """Vẽ một bông hoa Iris cách điệu: 3 cánh đứng + 3 cánh rủ + thân + lá."""
    up, down, streak, s, bg = STYLE[species]
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=130)
    ax.set_xlim(-3, 3); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.plot([0, 0.1], [-2.4, -0.2], c="#2e7d32", lw=5, solid_capstyle="round")   # thân
    ax.add_patch(Ellipse((-0.9, -1.5), 0.45, 1.8, angle=25, fc="#43a047", ec="none"))  # lá
    ax.add_patch(Ellipse((0.9, -1.5), 0.45, 1.8, angle=-25, fc="#388e3c", ec="none"))
    for ang in (90, 210, 330):          # 3 cánh đứng
        ax.add_patch(Ellipse((0, 0), 0.9 * s, 1.9 * s, angle=ang - 90, fc=up, ec="none", alpha=0.9))
    for ang in (270, 30, 150):          # 3 cánh rủ
        ax.add_patch(Ellipse((0, 0), 1.1 * s, 1.7 * s, angle=ang - 90, fc=down, ec="none", alpha=0.9))
        ax.add_patch(Ellipse((0, 0), 0.18 * s, 1.0 * s, angle=ang - 90, fc=streak, ec="none"))
    ax.add_patch(plt.Circle((0, 0), 0.18, fc="#fff3b0", ec="none"))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{species}.jpg"), facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    for sp in STYLE:
        draw(sp)
    print("Đã tạo ảnh minh họa trong images/")
