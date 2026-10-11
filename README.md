# Studio 2 - Klasifikasi Sampah

## Anggota

- Doni Agus Setiawan - 123140009
- Handialrizky       - 123140012
- Mega Zayyani       - 123140180

## Pendekatan

Model utama kelompok menggunakan backbone pretrained ImageNet dari empat kandidat: DenseNet-121, MobileNetV3-Large, EfficientNet-B0, dan ConvNeXt-Tiny. Training memakai augmentasi ringan, split validasi stratified, dan pemilihan bobot berdasarkan Macro-F1. Default submission adalah EfficientNet-B0 karena relatif ringan untuk batas waktu evaluasi.

Urutan kelas mengikuti `ImageFolder`:

1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

Tidak ada data tambahan dari luar pada baseline ini.

## Struktur data

Letakkan dataset pada folder `data/train/` dengan lima subfolder kelas sesuai urutan di atas. Dataset tidak boleh diunggah ke repository publik.

## Menjalankan

```powershell
python train.py --data data/train --model efficientnet_b0 --epochs 12
python evaluate.py --submission submission --data my_val
```

Untuk membandingkan kandidat lain, gunakan `--model densenet121`, `--model mobilenet_v3_large`, atau `--model convnext_tiny`. Model terbaik harus dilatih dengan `efficientnet_b0` sebelum disimpan sebagai `submission/model.pth`, karena `submission/model.py` memakai arsitektur default tersebut. Sebelum dikumpulkan, pastikan `submission/model.py` dan bobot dapat dijalankan pada environment evaluator.
