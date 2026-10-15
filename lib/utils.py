"""Tiện ích chung: vẽ lưới, lưu figure và sinh ảnh DEMO dự phòng."""
import math

import cv2
import matplotlib
matplotlib.use('Agg')  # chạy headless, không cần cửa sổ hiển thị
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, path, dpi=150):
    """Lưu figure ra file và đóng lại để tránh rò bộ nhớ."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def show_grid(items, columns=3, save_name=None, figsize=None):
    """Hiển thị danh sách (title, image) theo lưới, tùy chọn lưu figure."""
    rows = math.ceil(len(items) / columns)
    figsize = figsize or (6 * columns, 4.2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes, items):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    for ax in axes[len(items):]:
        ax.axis('off')
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    else:
        plt.close(fig)


def make_demo_gradient_pair(height=400, width=600):
    """Ảnh nền hoàng hôn tối + ảnh nguồn quả cầu sáng (kèm mask)."""
    y, x = np.mgrid[0:height, 0:width]
    bg = np.zeros((height, width, 3), np.uint8)
    bg[..., 0] = np.clip(40 + 90 * (y / height), 0, 255)
    bg[..., 1] = np.clip(60 + 50 * (y / height), 0, 255)
    bg[..., 2] = np.clip(120 - 60 * (y / height), 0, 255)
    bg[int(height * 0.65):] = (30, 90, 60)
    noise = np.random.default_rng(1).normal(0, 6, bg.shape)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    cv2.putText(bg, 'DEMO background', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 0, 255), 2, cv2.LINE_AA)

    sh, sw = 260, 300
    src = np.full((sh, sw, 3), (235, 235, 245), np.uint8)
    yy, xx = np.mgrid[0:sh, 0:sw]
    r = np.sqrt((xx - sw / 2) ** 2 + (yy - sh / 2) ** 2)
    ball = np.stack([255 - r * 0.9, 170 - r * 0.6, 40 + r * 0.3], -1)
    disk = r < 100
    src[disk] = np.clip(ball[disk], 0, 255).astype(np.uint8)
    cv2.putText(src, 'DEMO', (95, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 0, 255), 2, cv2.LINE_AA)
    mask = (disk * 255).astype(np.uint8)
    return bg, src, mask


def make_demo_scene(seed, width=1500, height=520):
    """Cảnh rộng nhiều texture để SIFT/ORB có đủ đặc trưng."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    scene = np.stack([120 + 60 * np.sin(x / 210.0 + seed),
                      140 + 50 * np.cos(y / 90.0),
                      150 + 40 * np.sin((x + y) / 260.0)], -1)
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    for _ in range(420):
        c = tuple(int(v) for v in rng.integers(20, 240, 3))
        px, py = int(rng.integers(0, width)), int(rng.integers(0, height))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(scene, (px, py), int(rng.integers(6, 34)), c, -1)
        elif kind == 1:
            cv2.rectangle(scene, (px, py),
                          (px + int(rng.integers(10, 60)), py + int(rng.integers(10, 60))), c, -1)
        else:
            cv2.line(scene, (px, py),
                     (px + int(rng.integers(-60, 60)), py + int(rng.integers(-60, 60))), c, 2)
    for i in range(30):
        cv2.putText(scene, f'{seed}-{i}',
                    (int(rng.integers(0, width - 80)), int(rng.integers(20, height))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    scene = np.clip(scene.astype(np.float32) + rng.normal(0, 5, scene.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(scene, (3, 3), 0)


def make_demo_panorama(seed, n_images, view_w=640, step=380):
    """Cắt các khung nhìn chồng lấn (~40%) từ một cảnh rộng."""
    scene = make_demo_scene(seed, width=step * (n_images - 1) + view_w + 60)
    imgs = []
    for i in range(n_images):
        x0 = 30 + i * step
        crop = scene[:, x0:x0 + view_w].copy()
        h, w = crop.shape[:2]
        jitter = np.random.default_rng(seed * 10 + i).uniform(-12, 12, (4, 2)).astype(np.float32)
        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        crop = cv2.warpPerspective(crop, cv2.getPerspectiveTransform(src_pts, src_pts + jitter),
                                   (w, h), borderMode=cv2.BORDER_REFLECT)
        crop = np.clip(crop.astype(np.float32) * (0.9 + 0.08 * i), 0, 255).astype(np.uint8)
        cv2.putText(crop, f'DEMO {seed}-{i + 1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 255), 2, cv2.LINE_AA)
        imgs.append(crop)
    return imgs
