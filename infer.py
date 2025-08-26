import argparse
import os
from typing import Tuple

import numpy as np
import torch
from gyro_temp_comp.models import UNet1D
from gyro_temp_comp.utils.checkpoint import load_checkpoint
from gyro_temp_comp.utils.signal import create_windows_1d, overlap_add_reconstruct, normalize_minmax, denormalize_minmax


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	data = np.load(path)
	gyro = np.asarray(data.get("gyro"), dtype=np.float32).reshape(-1)
	temp = np.asarray(data.get("temp"), dtype=np.float32).reshape(-1)
	target = data.get("target")
	if target is None:
		target = gyro.copy()
	else:
		target = np.asarray(target, dtype=np.float32).reshape(-1)
	T = min(len(gyro), len(temp), len(target))
	return gyro[:T], temp[:T], target[:T]


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Inference for gyro temp compensation")
	p.add_argument("--ckpt", required=True, type=str)
	p.add_argument("--input", required=True, type=str)
	p.add_argument("--output", required=True, type=str)
	p.add_argument("--window-size", type=int, default=2048)
	p.add_argument("--hop-size", type=int, default=1024)
	p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--num-downsamples", type=int, default=6)
	p.add_argument("--base-channels", type=int, default=64)
	p.add_argument("--use-instance-norm", action="store_true")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device(args.device)

	gyro, temp, target = load_npz(args.input)
	T = len(gyro)

	gmin, gmax = float(np.min(gyro)), float(np.max(gyro))
	tmin, tmax = float(np.min(temp)), float(np.max(temp))
	gyro_n = normalize_minmax(gyro, gmin, gmax)
	temp_n = normalize_minmax(temp, tmin, tmax)

	# Windowing
	sig = np.stack([gyro_n, temp_n], axis=0)  # (2, T)
	windows, window_fn = create_windows_1d(sig, args.window_size, args.hop_size, apply_hann=False)

	# Model
	required_multiple = 2 ** args.num_downsamples
	if args.window_size % required_multiple != 0:
		raise ValueError(f"window-size must be divisible by {required_multiple}")

	G = UNet1D(
		input_channels=2,
		output_channels=1,
		base_channels=args.base_channels,
		num_downsamples=args.num_downsamples,
		use_instance_norm=args.use_instance_norm,
	).to(device)

	ckpt = load_checkpoint(args.ckpt)
	G.load_state_dict(ckpt["model_state"])  # type: ignore[index]
	G.eval()

	outs = []
	with torch.no_grad():
		for w in windows:
			inp = torch.from_numpy(w[None, ...]).to(device)  # (1, 2, W)
			y = G(inp)
			outs.append(y.detach().cpu().numpy()[0, 0])
	outs = np.stack(outs, axis=0)  # (N, W)
	outs = outs[:, None, :]  # (N, 1, W) for overlap_add
	comp_norm = overlap_add_reconstruct(outs, args.hop_size, length=T, window_fn=window_fn)[0]
	comp = denormalize_minmax(comp_norm, gmin, gmax)

	os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
	np.savez_compressed(args.output, gyro=gyro, temp=temp, comp=comp, target=target)
	print(f"Saved: {args.output}")


if __name__ == "__main__":
	main()