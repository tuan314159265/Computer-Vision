# CO3057 — BTL2: Gradient Domain Editing & Image Stitching (Panorama)

Bài tập lớn môn Computer Vision. Notebook `CO3057_BTL2_Gradient_Panorama.ipynb` triển khai hai phần chính:

| Phần | Nội dung |
|---|---|
| **Phần 1** | Gradient Domain Editing (Poisson Blending) và so sánh với ghép trực tiếp (baseline) + `cv2.seamlessClone` |
| **Phần 2** | Ghép ảnh toàn cảnh dựa trên đặc trưng **SIFT** và **ORB** trên cùng một pipeline, chạy trên nhiều bộ ảnh |

## Chạy trên Google Colab

1. Mở notebook trong Colab.
2. Mục 1 (`USE_GOOGLE_DRIVE`): đặt `True` nếu dùng Google Drive (`MyDrive/DIPCV_BTL2/`), `False` nếu upload ảnh trực tiếp vào runtime.
3. Đặt ảnh theo cấu trúc thư mục bên dưới.
4. **Runtime → Run all**.

> Notebook **tự chứa toàn bộ code** nên `Run all` không phụ thuộc biến hay file cục bộ nào khác. Nếu thiếu ảnh thật, notebook tự sinh ảnh **DEMO** (có nhãn rõ ràng) để kiểm tra kỹ thuật — kết quả DEMO **không dùng để nộp**.

## Cấu trúc dữ liệu

```
DIPCV_BTL2/
└── data/
    ├── gradient/
    │   ├── background.jpg     # ảnh nền
    │   ├── source.jpg         # ảnh chứa đối tượng cần ghép
    │   └── mask.png           # (tuỳ chọn) mask trắng-đen; thiếu sẽ dùng GrabCut
    ├── pano_set1/             # bộ ảnh mẫu của đề (3–4 ảnh, tên theo thứ tự TRÁI → PHẢI)
    │   ├── 01.jpg  02.jpg  03.jpg ...
    └── pano_set2/             # bộ ảnh tự thu thập (cùng quy ước)
        ├── 01.jpg  02.jpg  03.jpg ...
```

Ảnh panorama được sắp theo **tên file**; thứ tự tên file = thứ tự trái → phải.

## Pipeline ghép ảnh (Phần 2)

```
Ảnh rời → Tiền xử lý → Feature detection & description → Feature matching
       → Lọc Lowe ratio → Homography + RANSAC → Warp về ảnh tham chiếu
       → Blending vùng chồng lấn (overwrite/average/feather) → Crop viền đen → Panorama
```

## Cấu trúc repository

```
CO3057_BTL2_Gradient_Panorama.ipynb   # notebook chính (tự chứa, chạy Run All)
lib/
  __init__.py
  gradient_domain.py                  # Poisson blending + tiện ích (bản tách riêng để tham khảo)
  panorama.py                         # pipeline SIFT/ORB (bản tách riêng để tham khảo)
data/
  README.md                           # hướng dẫn đặt ảnh
requirements.txt                      # cho người muốn chạy local (Colab đã cài sẵn)
README.md
```

`lib/` chứa các đoạn code dài đã tách ra khỏi notebook để báo cáo/review dễ đọc; nội dung tương đương code trong notebook.

## Kết quả đầu ra

Chạy xong, kết quả được ghi vào `outputs/`:

- Ảnh: panorama của từng bộ ảnh × từng phương pháp (`pano_<set>_sift.jpg`, `pano_<set>_orb.jpg`).
- Bảng số liệu (CSV): `part1_gradient_metrics.csv`, `part2_experiment_summary.csv`, `part2_pair_details.csv`.

## Yêu cầu môi trường (chạy local, không bắt buộc)

Colab đã cài sẵn các thư viện này. Chạy local:

```bash
pip install -r requirements.txt
```

SIFT cần `opencv-python >= 4.4.0` (bằng sáng chế đã hết hạn).
