"""Gradient Domain Editing — Poisson Blending.

Giải hệ Poisson thưa cho từng kênh màu để dán vùng đối tượng vào ảnh nền
sao cho gradient khớp ảnh nguồn còn biên khớp ảnh nền (không lộ đường nối).

Nội dung tương đương phần 1 của notebook.
"""
import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = [
    'resize_if_needed',
    'load_rgb',
    'list_images',
    'save_rgb',
    'direct_paste',
    'crop_to_mask',
    'mask_from_grabcut',
    'clean_mask',
    'poisson_blend',
    'seam_discontinuity',
    'seam_edge_energy',
    'object_color_shift',
]

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def resize_if_needed(rgb, max_side=1000):
    """Giảm ảnh quá lớn nhưng giữ nguyên tỉ lệ khung hình."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return rgb


def load_rgb(path, max_side=1000):
    """Đọc ảnh bằng OpenCV (BGR) và trả về RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Không đọc được ảnh: {path}')
    return resize_if_needed(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side)


def list_images(folder):
    """Liệt kê ảnh trong thư mục, sắp xếp theo tên file."""
    from pathlib import Path
    return sorted(p for p in Path(folder).glob('*') if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def direct_paste(bg_rgb, src_crop, mask_crop, offset):
    """Baseline: copy trực tiếp pixel trong mask."""
    out = bg_rgb.copy()
    y, x = offset
    h, w = mask_crop.shape
    region = out[y:y + h, x:x + w]
    region[mask_crop > 0] = src_crop[mask_crop > 0]
    return out


def crop_to_mask(src_rgb, mask, pad=2):
    """Cắt ảnh nguồn/mask theo bounding box của mask (+ viền pad)."""
    ys, xs = np.where(mask > 0)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return src_rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def mask_from_grabcut(rgb, rect, iters=6):
    """GrabCut khởi tạo bằng hình chữ nhật."""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def clean_mask(mask):
    """Giữ thành phần liên thông lớn nhất, lấp lỗ, làm mượt biên."""
    mask = (mask > 127).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = ((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def build_mask(src_rgb, mode='auto', mask_path=None, src_rect=None):
    """Tạo mask vùng đối tượng: ưu tiên file mask -> GrabCut -> ellipse."""
    from pathlib import Path
    h, w = src_rgb.shape[:2]
    if mode in ('auto', 'file') and mask_path is not None and Path(mask_path).exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            return clean_mask(m), 'file mask'
    if mode in ('auto', 'grabcut'):
        rect = src_rect or (int(0.05 * w), int(0.05 * h), int(0.9 * w), int(0.9 * h))
        return clean_mask(mask_from_grabcut(src_rgb, rect)), f'GrabCut rect={rect}'
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 5, h // 2 - 5), 0, 0, 360, 255, -1)
    return m, 'ellipse'


def poisson_blend(bg_rgb, src_crop, mask_crop, offset, mixing=False):
    """
    Poisson blending bằng sparse linear solver.
    mixing=False : trường hướng dẫn bằng gradient ảnh nguồn.
    mixing=True  : lấy gradient có độ lớn lớn hơn giữa nguồn và nền.
    """
    y0, x0 = offset
    h, w = mask_crop.shape
    bg = bg_rgb.astype(np.float64)
    src = src_crop.astype(np.float64)
    inside = mask_crop > 0

    idx = -np.ones((h, w), np.int64)
    ys, xs = np.where(inside)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)

    rows, cols, vals = [], [], []
    b = np.zeros((n, 3))
    for k, (y, x) in enumerate(zip(ys, xs)):
        rows.append(k)
        cols.append(k)
        vals.append(4.0)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            qy, qx = y + dy, x + dx
            inside_q = 0 <= qy < h and 0 <= qx < w and inside[qy, qx]
            g_src = src[y, x] - src[qy, qx] if inside_q else np.zeros(3)
            g_bg = bg[y0 + y, x0 + x] - bg[y0 + qy, x0 + qx]
            if mixing:
                v = g_src if np.abs(g_src).sum() > np.abs(g_bg).sum() else g_bg
            else:
                v = g_src
            b[k] += v
            if 0 <= qy < h and 0 <= qx < w and inside[qy, qx]:
                rows.append(k)
                cols.append(idx[qy, qx])
                vals.append(-1.0)
            else:
                b[k] += bg[y0 + qy, x0 + qx]

    A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    lu = splu(A)
    f = np.stack([lu.solve(b[:, c]) for c in range(3)], axis=1)

    out = bg.copy()
    out[y0 + ys, x0 + xs] = f
    return np.clip(out, 0, 255).astype(np.uint8)


def seam_discontinuity(result_rgb, mask_crop, offset):
    """Sai khác trung bình giữa vành pixel trong mask sát biên và vành pixel ngoài mask."""
    y0, x0 = offset
    h, w = mask_crop.shape
    full = np.zeros(result_rgb.shape[:2], np.uint8)
    full[y0:y0 + h, x0:x0 + w] = mask_crop
    kernel = np.ones((3, 3), np.uint8)
    inner = (full > 0) & (cv2.erode(full, kernel) == 0)
    outer = (cv2.dilate(full, kernel) > 0) & (full == 0)
    if inner.sum() == 0 or outer.sum() == 0:
        return float('nan')
    return float(np.abs(result_rgb[inner].astype(np.float64).mean(0) - result_rgb[outer].astype(np.float64).mean(0)).mean())


def seam_edge_energy(result_rgb, mask_crop, offset):
    """Độ lớn gradient trung bình dọc biên mask (gradient giả do đường nối)."""
    y0, x0 = offset
    h, w = mask_crop.shape
    full = np.zeros(result_rgb.shape[:2], np.uint8)
    full[y0:y0 + h, x0:x0 + w] = mask_crop
    ring = cv2.morphologyEx(full, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return float(np.sqrt(gx ** 2 + gy ** 2)[ring].mean())


def object_color_shift(result_rgb, src_crop, mask_crop, offset):
    """Độ lệch trung bình của màu đối tượng so với ảnh nguồn gốc."""
    y0, x0 = offset
    h, w = mask_crop.shape
    region = result_rgb[y0:y0 + h, x0:x0 + w].astype(np.float64)
    m = mask_crop > 0
    return float(np.abs(region[m] - src_crop.astype(np.float64)[m]).mean())
