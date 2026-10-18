# Window AC Condenser Dent AI — V2

Pipeline dua tahap:

1. **Detector** menemukan area kondensor di frame kamera dan menggambar kotak (bounding box).
2. **Classifier** (MobileNetV3, model lama lu) hanya menilai isi kotak itu: `OK` atau `NG` (dent).

Status di layar: `UNDETECTED` (default, tidak ada kondensor) -> `ANALYZING` -> `OK` / `NG` / `REVIEW`.
Penghitung: total, jumlah OK, NG, REVIEW, dan persentasenya. Satu unit dihitung **sekali**
(hasil dirata-rata dari beberapa frame, dikunci, dan baru dilepas setelah kondensor hilang dari frame).

## Instalasi
Tidak ada dependency baru. Copy semua file V2 ke folder project lama (timpa file yang sama; `.venv`, `dataset_raw`, `data`, `models` aman).

```bash
source .venv/bin/activate
```

## Alur kerja

### A. Jalan langsung (tanpa training detector)
Detector heuristik (mencari pola sirip paralel) aktif otomatis kalau model detector belum ada.

```bash
python prepare_dataset.py                 # crop area kondensor -> data/train|val|test
python train.py --epochs 12 --batch-size 16
python live_inference.py --camera 0
```

### B. Detector yang lebih akurat (disarankan sebelum dipakai serius)
```bash
python capture_dataset.py      # tombol b = foto background TANPA kondensor (ambil 20+)
python label_boxes.py          # konfirmasi/gambar kotak kondensor; Enter = terima usulan kuning
python train_detector.py --epochs 30
python prepare_dataset.py      # ulang, sekarang memakai kotak manual + detector hasil training
python train.py --epochs 12 --batch-size 16
python live_inference.py --camera 0
```
Kotak manual di `dataset_raw/boxes.json` selalu diprioritaskan. Detector hasil training dipakai otomatis
(`--detector heuristic|learned|auto`).

## Kontrol live_inference
`q` keluar | `s` snapshot | `n` periksa ulang unit yang sedang terlihat | `r` reset counter sesi

Opsi penting:
- `--confidence 0.75` di bawah ini hasil jadi `REVIEW` (perlu cek manusia)
- `--vote-frames 10` jumlah frame yang dirata-rata per unit
- `--min-frames 5` tunggu kondensor stabil dulu
- `--lost-frames 15` jumlah frame tanpa kondensor sebelum unit dianggap sudah pergi (kembali ke UNDETECTED)
- `--resume` lanjutkan counter dari total hari ini di database
- `--detect-every 2` kalau laptop lambat

## Dashboard
```bash
streamlit run dashboard.py
```
Menampilkan total, OK, NG, REVIEW + persentase, pie chart, unit per jam, NG rate per jam, riwayat, bukti gambar.
Database lama (V1) otomatis dimigrasi; data V1 bisa dilihat lewat pilihan "Legacy frame samples".

## Catatan validasi
- Detector heuristik bisa salah mendeteksi pemandangan lain yang bergaris-garis (mis. lantai, rak). Untuk lini produksi pakai detector hasil training dengan foto background.
- Pastikan foto GOOD dan DENT punya gaya yang sama (kamera, sudut, cahaya). Kalau tidak, model bisa belajar perbedaan gaya foto, bukan dent.
- Foto dari webcam yang mengarah ke layar HP/monitor berbeda dari foto kondensor langsung; sebaiknya ganti dengan foto langsung.
- Evaluasi di `data/test` (terutama NG recall) sebelum hasil dipercaya. Jangan jadikan interlock/auto-reject di V1.
