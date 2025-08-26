import os
import argparse
import numpy as np


def synthesize_sample(T: int, seed: int) -> dict:
	rng = np.random.default_rng(seed)
	t = np.linspace(0, 1, T, endpoint=False)
	# Temperature: slow varying
	temp = 25 + 5 * np.sin(2 * np.pi * 0.2 * t) + 0.3 * rng.standard_normal(T)
	# Gyro raw: fast varying + temperature-dependent bias/drift
	fast_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
	temp_effect = 0.02 * (temp - 25.0) + 0.01 * (temp - 25.0) ** 2
	gyro = fast_signal + temp_effect + 0.05 * rng.standard_normal(T)
	# Target compensated gyro ideally removes temp_effect
	target = fast_signal + 0.05 * rng.standard_normal(T)
	return {"gyro": gyro.astype(np.float32), "temp": temp.astype(np.float32), "target": target.astype(np.float32)}


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--out", required=True, type=str)
	p.add_argument("--num-train", type=int, default=128)
	p.add_argument("--num-val", type=int, default=32)
	p.add_argument("--length", type=int, default=4096)
	args = p.parse_args()

	train_dir = os.path.join(args.out, "train")
	val_dir = os.path.join(args.out, "val")
	os.makedirs(train_dir, exist_ok=True)
	os.makedirs(val_dir, exist_ok=True)

	for i in range(args.num_train):
		data = synthesize_sample(args.length, seed=1000 + i)
		path = os.path.join(train_dir, f"sample_{i:03d}.npz")
		np.savez_compressed(path, **data)
	print(f"Wrote {args.num_train} training samples to {train_dir}")

	for i in range(args.num_val):
		data = synthesize_sample(args.length, seed=2000 + i)
		path = os.path.join(val_dir, f"sample_{i:03d}.npz")
		np.savez_compressed(path, **data)
	print(f"Wrote {args.num_val} validation samples to {val_dir}")


if __name__ == "__main__":
	main()