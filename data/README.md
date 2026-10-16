# Dữ liệu đầu vào

Đặt ảnh theo cấu trúc dưới đây. Tên file được sắp xếp theo thứ tự từ trái sang phải.

```text
data/
├── gradient/
│   ├── background.jpg
│   ├── source.jpg
│   └── mask.png              # tuỳ chọn; thiếu thì chương trình dùng GrabCut
├── pano_set1/                # ảnh boat
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
├── pano_set2/                # ảnh Budapest
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
└── test_assets/              # ảnh kiểm thử cục bộ; không commit lên Git
    ├── boat1.jpg ... boat6.jpg
    ├── budapest1.jpg ... budapest6.jpg
    └── gradient/
```

Mỗi panorama set cần ít nhất hai ảnh. Nếu `pano_set1/` hoặc `pano_set2/` chưa có đủ ảnh, chương trình thử dùng lần lượt `boat*.jpg` hoặc `budapest*.jpg` trong `test_assets/`. Nếu vẫn không có ảnh phù hợp, chương trình sinh ảnh DEMO để kiểm tra kỹ thuật.

Ảnh trong `test_assets/` và `pano_set2/` hiện bị `.gitignore` loại khỏi Git để tránh đẩy dữ liệu ảnh lên remote. Khi clone repo, tự chép/cung cấp dữ liệu vào đúng vị trí.
