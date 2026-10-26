# CO3057 — Gradient Domain Editing & Panorama Stitching

Project computer vision gồm hai phần:

1. **Gradient Domain Editing** — ghép đối tượng bằng Poisson blending; so sánh với copy-paste và `cv2.seamlessClone`.
2. **Panorama stitching** — một pipeline dùng SIFT và ORB: matching, Lowe ratio test, homography với RANSAC, warping và feather blending.

Nếu một cặp ảnh không có đủ inlier để ước lượng homography ổn định, chương trình không tích lũy homography yếu (nguyên nhân gây panorama bị kéo/loe): trước tiên tìm lại thứ tự ảnh theo các vùng chồng lấn đáng tin cậy. Với chuỗi ảnh phẳng được sắp lại, dùng OpenCV SCANS để chọn seam; nếu không tìm được chuỗi đáng tin cậy, dùng OpenCV Stitcher làm fallback. Backend và thứ tự ảnh được ghi trong `part2_experiment_summary.csv`.

Mã chạy local đã được tách thành các module Python trong `lib/`.

## Bắt đầu nhanh

Yêu cầu: Python 3.10+ và pip.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Trên Windows PowerShell, kích hoạt môi trường bằng `.venv\Scripts\Activate.ps1`.

Có thể đổi thư mục dữ liệu hoặc kết quả:

```bash
python main.py --data-dir data --output-dir outputs
```

Chương trình chạy hoàn toàn local, không cần Google Drive. Nếu thiếu đầu vào bắt buộc, chương trình tạo ảnh DEMO có nhãn để kiểm tra pipeline; kết quả DEMO không dùng để nộp.

## Cấu trúc dữ liệu

Tên file trong mỗi panorama set được sắp theo thứ tự từ trái sang phải.

```text
data/
├── gradient/
│   ├── background.jpg
│   ├── source.jpg
│   └── mask.png                 # tuỳ chọn; thiếu thì dùng GrabCut
├── pano_set1/                   # ảnh boat, tối thiểu 2 ảnh
│   ├── boat1.jpg
│   ├── boat2.jpg
│   └── ...
├── pano_set2/                   # ảnh Budapest, tối thiểu 2 ảnh
│   ├── budapest1.jpg
│   ├── budapest2.jpg
│   └── ...
└── test_assets/                 # dữ liệu tham khảo thêm; không đưa lên Git
    ├── boat1.jpg ...
    ├── budapest1.jpg ...
    └── gradient/
```

Loader tìm dữ liệu theo thứ tự:

- Gradient: `data/gradient/`, sau đó `data/test_assets/gradient/`.
- `pano_set1`: `data/pano_set1/`, sau đó ảnh `boat*.jpg` trong `data/test_assets/`.
- `pano_set2`: `data/pano_set2/`, sau đó ảnh `budapest*.jpg` trong `data/test_assets/`.

Repo bao gồm sáu ảnh boat trong `data/pano_set1/`, sáu ảnh Budapest trong `data/pano_set2/`, cùng cặp ảnh gradient. Thư mục `gradient-20261016T073240Z-1-001/` bạn chỉ ra đang rỗng, nên chưa có file `gadinest` nào để thêm. Nếu đó là ảnh/bộ ảnh bổ sung, hãy đặt ảnh vào đúng chuỗi trong `data/pano_set2/`.

Chỉ `data/test_assets/` bị loại khỏi Git; hai bộ ảnh và cặp gradient dùng trong bài được theo dõi trong repo để có thể chạy ngay sau khi clone.

## Kết quả

Chạy xong, kết quả được lưu trong `outputs/`:

- `part1_gradient_metrics.csv` — chỉ số đánh giá ghép ảnh phần 1.
- `part1_comparison.png` — hình so sánh các phương pháp phần 1.
- `part2_experiment_summary.csv` — tổng kết SIFT/ORB theo bộ ảnh.
- `part2_pair_details.csv` — số liệu theo từng cặp ảnh.
- `pano_<set>_<method>.jpg` — panorama, ví dụ `pano_pano_set1_sift.jpg`.

Các kết quả chính trong `outputs/` được lưu trong repo; kết quả phụ và ảnh DEMO vẫn bị `.gitignore` loại trừ.

## Cấu trúc mã nguồn

```text
main.py                    # CLI / điểm chạy chính
lib/
├── data.py                # tìm và nạp ảnh thật hoặc DEMO
├── experiments.py         # điều phối hai phần và xuất CSV/ảnh
├── gradient_domain.py     # mask, Poisson blending, chỉ số biên
├── panorama.py            # pipeline SIFT/ORB
└── utils.py               # vẽ và sinh ảnh DEMO
requirements.txt
```

## Git

Repo công khai: [github.com/tuan314159265/Computer-Vision](https://github.com/tuan314159265/Computer-Vision). Mã nguồn, dữ liệu đầu vào dùng trong bài và các kết quả chính đều được push lên nhánh `master`. Notebook gốc và thư mục `data/test_assets/` vẫn bị ignore.
