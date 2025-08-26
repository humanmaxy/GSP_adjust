### Gyro Temperature Compensation with GAN (1D U-Net Generator)



#### 训练
```bash
python train.py \
  --train-glob "/workspace/data_dummy/train/*.npz" \
  --val-glob "/workspace/data_dummy/val/*.npz" \
  --epochs 20 --batch-size 16 --lr 2e-4 --lambda-l1 100 \
  --save-dir /workspace/checkpoints
```

#### 推理
```bash
python infer.py \
  --ckpt /workspace/checkpoints/best.pt \
  --input "/workspace/data_dummy/val/sample_000.npz" \
  --output "/workspace/out_sample_000.npz" \
  --window-size 2048 --hop-size 1024
```

Data files are `.npz` with keys: `gyro`, `temp`, `target` (optional; if missing, the dataset will compute `target = gyro` as identity).
npy数据输入是陀螺仪测量值gyro、温度temp和实际转速target
