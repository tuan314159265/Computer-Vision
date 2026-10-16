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

Hai bộ ảnh boat và Budapest hiện đã có sẵn trong repo. Mỗi panorama set cần ít nhất hai ảnh. Nếu sau này thay hoặc xoá ảnh trong `pano_set1/` hay `pano_set2/`, chương trình thử dùng lần lượt `boat*.jpg` hoặc `budapest*.jpg` trong `test_assets/`; nếu không có đủ ảnh, chương trình sinh ảnh DEMO để kiểm tra kỹ thuật.

Các ảnh bài làm trong `gradient/`, `pano_set1/` và `pano_set2/` được theo dõi trong Git. `test_assets/` chỉ là dữ liệu tham khảo bổ sung và bị `.gitignore` loại khỏi repo.
