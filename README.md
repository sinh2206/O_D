# O_D

## 1. Cài môi trường trên Google Colab

```bash

!pip install -r requirements.txt
```

Khuyến nghị:

- vào `Runtime` -> `Change runtime type`
- chọn `GPU`

## 2. Cách huấn luyện

```bash
!python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models
```

Sau khi train:

- mô hình tốt nhất: `models/best.pth`
- mô hình gần nhất: `models/last.pth`

## 3. Cách chạy suy luận

```bash
!python predict.py \
  --image_dir ./public/val/images \
  --val_annotation ./public/annotations/val.json \
  --output ./val_predictions.json \
  --results_dir ./results \
  --checkpoint ./models/best.pth
```

Kết quả sinh ra:

- file dự đoán: `val_predictions.json`
- file hardcase: `results/hardcase_summary.json`

## 4. Vị trí đặt mô hình / trọng số mô hình

Đặt file trọng số trong thư mục:

```text
models/
```

Ví dụ:

- `models/best.pth`
- `models/last.pth`

Nếu bạn có trọng số riêng, chỉ cần chép vào `models/` rồi truyền đường dẫn qua `--checkpoint`.

Ví dụ:

```bash
!python predict.py \
  --image_dir ./public/val/images \
  --val_annotation ./public/annotations/val.json \
  --output ./val_predictions.json \
  --results_dir ./results \
  --checkpoint ./models/ten_trong_so_cua_ban.pth
```
