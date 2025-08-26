### Gyro Temperature Compensation with GAN (1D U-Net Generator)

This project trains a GAN to compensate gyro readings using temperature as an auxiliary slow-varying input. The generator is a 1D U-Net; the discriminator is a 1D PatchGAN. Input channels: `[gyro_raw_fast, temperature_slow]`. The model outputs the compensated gyro signal.

#### Install
```bash
pip install -r requirements.txt
```

#### Dummy data
```bash
python scripts/generate_dummy_data.py --out /workspace/data_dummy --num-train 128 --num-val 32 --length 4096
```

#### Train
```bash
python train.py \
  --train-glob "/workspace/data_dummy/train/*.npz" \
  --val-glob "/workspace/data_dummy/val/*.npz" \
  --epochs 20 --batch-size 16 --lr 2e-4 --lambda-l1 100 \
  --save-dir /workspace/checkpoints
```

#### Inference
```bash
python infer.py \
  --ckpt /workspace/checkpoints/best.pt \
  --input "/workspace/data_dummy/val/sample_000.npz" \
  --output "/workspace/out_sample_000.npz" \
  --window-size 2048 --hop-size 1024
```

Data files are `.npz` with keys: `gyro`, `temp`, `target` (optional; if missing, the dataset will compute `target = gyro` as identity).