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
#### 数据格式与样例
- 输入 `.npz` 键: `gyro`、`temp`、`target`（`target` 可选，缺省则使用 `gyro` 作为同一目标）
- 推理输出 `.npz` 键: `gyro`、`temp`、`comp`、`target`（`comp` 为补偿后的输出）
- 所有数组均为 `float32`，一维，长度一致。
- 文本样例 `.txt` 为空格分隔列：
  - 训练/输入: `gyro temp target`
  - 推理/输出: `gyro temp comp target`
  - 第一行含列说明，第二行含长度信息，其余为逐时刻数据。

样例路径
- 输入样例: `/workspace/data/samples/sample_small.npz`, `/workspace/data/samples/sample_small.txt`
- 输出样例: `/workspace/data/samples/sample_small_out.npz`, `/workspace/data/samples/sample_small_out.txt
