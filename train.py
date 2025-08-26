import os
import argparse
import random
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from gyro_temp_comp.models import UNet1D, PatchDiscriminator1D
from gyro_temp_comp.data.dataset import GyroTempWindowDataset
from gyro_temp_comp.utils.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def build_models(
	input_channels: int,
	output_channels: int,
	base_channels: int,
	num_downsamples: int,
	use_instance_norm: bool,
) -> Dict[str, Any]:
	generator = UNet1D(
		input_channels=input_channels,
		output_channels=output_channels,
		base_channels=base_channels,
		num_downsamples=num_downsamples,
		use_instance_norm=use_instance_norm,
	)
	discriminator = PatchDiscriminator1D(
		input_channels=input_channels,
		output_channels=output_channels,
		base_channels=base_channels,
	)
	return {"G": generator, "D": discriminator}


def train_one_epoch(
	generator: nn.Module,
	discriminator: nn.Module,
	loader: DataLoader,
	optimizer_g: torch.optim.Optimizer,
	optimizer_d: torch.optim.Optimizer,
	device: torch.device,
	lambda_l1: float,
	amp: bool,
) -> Dict[str, float]:
	bce_loss = nn.BCEWithLogitsLoss()
	l1_loss = nn.L1Loss()
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	generator.train()
	discriminator.train()

	loss_g_running = 0.0
	loss_d_running = 0.0
	loss_l1_running = 0.0
	count = 0

	pbar = tqdm(loader, desc="train", leave=False)
	for x, y in pbar:
		x = x.to(device)
		y = y.to(device)

		batch_size = x.shape[0]

		# ---------------------
		# Train Discriminator
		# ---------------------
		optimizer_d.zero_grad(set_to_none=True)
		with torch.cuda.amp.autocast(enabled=amp):
			fake_y = generator(x)
			pred_real = discriminator(x, y)
			pred_fake = discriminator(x, fake_y.detach())
			target_real = torch.ones_like(pred_real)
			target_fake = torch.zeros_like(pred_fake)
			loss_d_real = bce_loss(pred_real, target_real)
			loss_d_fake = bce_loss(pred_fake, target_fake)
			loss_d = 0.5 * (loss_d_real + loss_d_fake)
		scaler.scale(loss_d).backward()
		scaler.step(optimizer_d)

		# ---------------------
		# Train Generator
		# ---------------------
		optimizer_g.zero_grad(set_to_none=True)
		with torch.cuda.amp.autocast(enabled=amp):
			fake_y = generator(x)
			pred_fake_for_g = discriminator(x, fake_y)
			adv_loss = bce_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
			l1 = l1_loss(fake_y, y)
			loss_g = adv_loss + lambda_l1 * l1
		scaler.scale(loss_g).backward()
		scaler.step(optimizer_g)
		scaler.update()

		loss_g_running += float(loss_g.detach().cpu()) * batch_size
		loss_d_running += float(loss_d.detach().cpu()) * batch_size
		loss_l1_running += float(l1.detach().cpu()) * batch_size
		count += batch_size

		pbar.set_postfix({
			"loss_d": f"{loss_d_running / max(count,1):.4f}",
			"loss_g": f"{loss_g_running / max(count,1):.4f}",
			"l1": f"{loss_l1_running / max(count,1):.4f}",
		})

	return {
		"loss_d": loss_d_running / max(count, 1),
		"loss_g": loss_g_running / max(count, 1),
		"l1": loss_l1_running / max(count, 1),
	}


def validate(
	generator: nn.Module,
	loader: DataLoader,
	device: torch.device,
	amp: bool,
) -> Dict[str, float]:
	l1_loss = nn.L1Loss(reduction="sum")
	generator.eval()
	l1_sum = 0.0
	num = 0
	with torch.no_grad():
		for x, y in tqdm(loader, desc="val", leave=False):
			x = x.to(device)
			y = y.to(device)
			with torch.cuda.amp.autocast(enabled=amp):
				fake_y = generator(x)
				l1 = l1_loss(fake_y, y)
			l1_sum += float(l1.detach().cpu())
			num += int(np.prod(list(y.shape)))
	l1_mae = l1_sum / max(num, 1)
	return {"l1_mae": l1_mae}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train GAN for gyro temp compensation with 1D U-Net generator")
	parser.add_argument("--train-glob", required=True, type=str, help="Glob for training .npz files")
	parser.add_argument("--val-glob", required=True, type=str, help="Glob for validation .npz files")
	parser.add_argument("--window-size", type=int, default=2048)
	parser.add_argument("--hop-size", type=int, default=1024)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--beta1", type=float, default=0.5)
	parser.add_argument("--lambda-l1", type=float, default=100.0)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--save-dir", type=str, default="/workspace/checkpoints")
	parser.add_argument("--save-every", type=int, default=1)
	parser.add_argument("--val-every", type=int, default=1)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--num-downsamples", type=int, default=6)
	parser.add_argument("--base-channels", type=int, default=64)
	parser.add_argument("--use-instance-norm", action="store_true")
	parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	os.makedirs(args.save_dir, exist_ok=True)
	set_seed(args.seed)

	device = torch.device(args.device)

	required_multiple = 2 ** args.num_downsamples
	if args.window_size % required_multiple != 0:
		raise ValueError(f"window-size must be divisible by {required_multiple}")
	if args.hop_size <= 0 or args.hop_size > args.window_size:
		raise ValueError("hop-size must be > 0 and <= window-size")

	train_ds = GyroTempWindowDataset(
		files_glob=args.train_glob,
		window_size=args.window_size,
		hop_size=args.hop_size,
		require_multiple_of=required_multiple,
	)
	val_ds = GyroTempWindowDataset(
		files_glob=args.val_glob,
		window_size=args.window_size,
		hop_size=args.hop_size,
		require_multiple_of=required_multiple,
	)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	models = build_models(
		input_channels=2,
		output_channels=1,
		base_channels=args.base_channels,
		num_downsamples=args.num_downsamples,
		use_instance_norm=args.use_instance_norm,
	)
	G: nn.Module = models["G"].to(device)
	D: nn.Module = models["D"].to(device)

	optimizer_g = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
	optimizer_d = Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

	best_val = float("inf")
	start_epoch = 0

	if args.resume:
		ckpt = load_checkpoint(args.resume)
		G.load_state_dict(ckpt["model_state"])  # generator state
		if ckpt.get("optimizer_g_state"):
			optimizer_g.load_state_dict(ckpt["optimizer_g_state"])  # type: ignore[index]
		if ckpt.get("optimizer_d_state"):
			optimizer_d.load_state_dict(ckpt["optimizer_d_state"])  # type: ignore[index]
		start_epoch = int(ckpt.get("epoch", 0))
		best_val = float(ckpt.get("best_metric", best_val))

	meta = {
		"args": vars(args),
		"model_kwargs": {
			"input_channels": 2,
			"output_channels": 1,
			"base_channels": args.base_channels,
			"num_downsamples": args.num_downsamples,
			"use_instance_norm": args.use_instance_norm,
		},
	}

	for epoch in range(start_epoch, args.epochs):
		print(f"Epoch {epoch+1}/{args.epochs}")
		train_metrics = train_one_epoch(
			generator=G,
			discriminator=D,
			loader=train_loader,
			optimizer_g=optimizer_g,
			optimizer_d=optimizer_d,
			device=device,
			lambda_l1=args.lambda_l1,
			amp=args.amp,
		)
		print({k: f"{v:.6f}" for k, v in train_metrics.items()})

		if (epoch + 1) % args.val_every == 0:
			val_metrics = validate(G, val_loader, device=device, amp=args.amp)
			print({k: f"{v:.6f}" for k, v in val_metrics.items()})
			val_score = float(val_metrics["l1_mae"])  # lower is better
			if val_score < best_val:
				best_val = val_score
				best_path = os.path.join(args.save_dir, "best.pt")
				save_checkpoint(
					best_path,
					model_state=G.state_dict(),
					optimizer_g_state=optimizer_g.state_dict(),
					optimizer_d_state=optimizer_d.state_dict(),
					epoch=epoch + 1,
					best_metric=best_val,
					meta=meta,
				)
				print(f"Saved best checkpoint to {best_path} (l1_mae={best_val:.6f})")

		if (epoch + 1) % args.save_every == 0:
			latest_path = os.path.join(args.save_dir, "latest.pt")
			save_checkpoint(
				latest_path,
				model_state=G.state_dict(),
				optimizer_g_state=optimizer_g.state_dict(),
				optimizer_d_state=optimizer_d.state_dict(),
				epoch=epoch + 1,
				best_metric=best_val,
				meta=meta,
			)
			print(f"Saved latest checkpoint to {latest_path}")


if __name__ == "__main__":
	main()