"""Fast predictor evaluation on a limited validation subset.

This utility mirrors the validation logic from `train_predictor.py`
without requiring DDP. It loads the fine-tuned tokenizer/predictor
specified in the config (or CLI overrides) and reports loss/accuracy
metrics on up to `config.n_fast_test_iter` samples for a quick health check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure project root is importable when running from finetune/
sys.path.append('../')

from config import Config
from dataset import QlibDataset
from model.kronos import Kronos, KronosTokenizer
from utils.training_utils import cleanup_ddp, format_time, set_seed, setup_ddp


@dataclass
class FastTestArgs:
	predictor_path: str
	tokenizer_path: str
	device: torch.device
	max_samples: int
	batch_size: int
	num_workers: int
	log_interval: int
	output_json: Optional[str]


def _prepare_dataloader(
	config: Config,
	batch_size: int,
	num_workers: int,
	distributed: bool,
	rank: int,
	world_size: int,
) -> DataLoader:
	dataset = QlibDataset('val')
	dataset.set_epoch_seed(rank if distributed else 0)
	sampler = None
	if distributed:
		sampler = DistributedSampler(
			dataset,
			num_replicas=world_size,
			rank=rank,
			shuffle=False,
			drop_last=False,
		)

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False,
		sampler=sampler,
	)
	return loader


def _load_models(predictor_path: str, tokenizer_path: str, device: torch.device) -> tuple[Kronos, KronosTokenizer]:
	tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
	tokenizer.eval().to(device)

	model = Kronos.from_pretrained(predictor_path)
	model.eval().to(device)
	return model, tokenizer


@torch.inference_mode()
def run_fast_test(
	config: Config,
	args: FastTestArgs,
	rank: int = 0,
	world_size: int = 1,
	use_ddp: bool = False,
) -> Dict[str, Any]:
	global_max_samples = min(args.max_samples, config.n_val_iter)
	loader = _prepare_dataloader(
		config,
		args.batch_size,
		args.num_workers,
		distributed=use_ddp,
		rank=rank,
		world_size=world_size,
	)
	global_max_samples = min(global_max_samples, len(loader.dataset))
	per_rank_max_samples = math.ceil(global_max_samples / world_size)
	model, tokenizer = _load_models(args.predictor_path, args.tokenizer_path, args.device)

	totals = {
		'loss': 0.0,
		's1_loss': 0.0,
		's2_loss': 0.0,
		's1_correct': 0,
		's2_correct': 0,
		'tokens': 0,
		'samples': 0,
		'batches': 0,
	}

	start_time = time.time()

	for batch_idx, (batch_x, batch_x_stamp) in enumerate(loader):
		remaining = per_rank_max_samples - totals['samples']
		if remaining <= 0:
			break

		if batch_x.size(0) > remaining:
			batch_x = batch_x[:remaining]
			batch_x_stamp = batch_x_stamp[:remaining]

		batch_x = batch_x.to(args.device, non_blocking=True)
		batch_x_stamp = batch_x_stamp.to(args.device, non_blocking=True)

		token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
		token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
		token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
		logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
		loss, s1_loss, s2_loss = model.head.compute_loss(
			logits[0], logits[1], token_out[0], token_out[1]
		)

		batch_tokens = token_out[0].numel()
		totals['loss'] += loss.item() * batch_tokens
		totals['s1_loss'] += s1_loss.item() * batch_tokens
		totals['s2_loss'] += s2_loss.item() * batch_tokens
		totals['tokens'] += batch_tokens

		s1_preds = logits[0].argmax(dim=-1)
		s2_preds = logits[1].argmax(dim=-1)
		totals['s1_correct'] += (s1_preds == token_out[0]).sum().item()
		totals['s2_correct'] += (s2_preds == token_out[1]).sum().item()

		totals['samples'] += batch_x.size(0)
		totals['batches'] += 1

		if (
			((batch_idx + 1) % args.log_interval == 0) or totals['samples'] >= per_rank_max_samples
		) and (not use_ddp or rank == 0):
			avg_loss = totals['loss'] / max(1, totals['tokens'])
			log_prefix = f"[FastTest][Rank {rank}]" if use_ddp else "[FastTest]"
			print(
				f"{log_prefix} Processed {totals['samples']}/{per_rank_max_samples} local samples | "
				f"Avg Loss: {avg_loss:.4f} | S1 Acc: {totals['s1_correct'] / max(1, totals['tokens']):.4f} | "
				f"S2 Acc: {totals['s2_correct'] / max(1, totals['tokens']):.4f}"
			)

	if args.device.type == 'cuda':
		torch.cuda.synchronize(args.device)
	elapsed = time.time() - start_time

	metrics_tensor = torch.tensor([
		totals['loss'],
		totals['s1_loss'],
		totals['s2_loss'],
		float(totals['s1_correct']),
		float(totals['s2_correct']),
		float(totals['tokens']),
		float(totals['samples']),
		float(totals['batches']),
	], device=args.device, dtype=torch.float64)
	if use_ddp:
		dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

	elapsed_tensor = torch.tensor([elapsed], device=args.device, dtype=torch.float64)
	if use_ddp:
		dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

	(
		tot_loss,
		tot_s1_loss,
		tot_s2_loss,
		tot_s1_correct,
		tot_s2_correct,
		tot_tokens,
		tot_samples,
		tot_batches,
	) = metrics_tensor.tolist()
	elapsed = elapsed_tensor.item()

	avg_loss = tot_loss / max(1.0, tot_tokens)
	avg_s1_loss = tot_s1_loss / max(1.0, tot_tokens)
	avg_s2_loss = tot_s2_loss / max(1.0, tot_tokens)

	results = {
		'samples_evaluated': tot_samples,
		'tokens_evaluated': tot_tokens,
		'avg_loss': avg_loss,
		'avg_s1_loss': avg_s1_loss,
		'avg_s2_loss': avg_s2_loss,
		's1_accuracy': tot_s1_correct / max(1.0, tot_tokens),
		's2_accuracy': tot_s2_correct / max(1.0, tot_tokens),
		'perplexity': math.exp(avg_loss) if avg_loss < 20 else float('inf'),
		'batches_processed': tot_batches,
		'wall_time_sec': elapsed,
		'wall_time_hms': format_time(elapsed),
		'samples_per_second': tot_samples / max(elapsed, 1e-9),
		'tokens_per_second': tot_tokens / max(elapsed, 1e-9),
		'global_max_samples': global_max_samples,
		'per_rank_max_samples': per_rank_max_samples,
		'world_size': world_size,
	}

	if rank == 0:
		print("\n=== Fast Test Summary ===")
		for key, value in results.items():
			if isinstance(value, float):
				print(f"{key}: {value:.4f}")
			else:
				print(f"{key}: {value}")

		if args.output_json:
			with open(args.output_json, 'w') as f:
				json.dump(results, f, indent=4)
			print(f"Metrics saved to {args.output_json}")

	return results


def parse_cli(config: Config) -> tuple[FastTestArgs, argparse.Namespace]:
	parser = argparse.ArgumentParser(description="Quick predictor validation on a subset")
	parser.add_argument('--predictor_path', type=str, default=None, help='Override predictor checkpoint path')
	parser.add_argument('--tokenizer_path', type=str, default=None, help='Override tokenizer checkpoint path')
	parser.add_argument('--max_samples', type=int, default=None, help='Max samples to evaluate (defaults to config.n_fast_test_iter)')
	parser.add_argument('--batch_size', type=int, default=None, help='Batch size for evaluation (defaults to config.batch_size)')
	parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
	parser.add_argument('--log_interval', type=int, default=5, help='Print frequency in batches')
	parser.add_argument('--device', type=str, default=None, help='Device string, e.g. cuda:0 or cpu')
	parser.add_argument('--output_json', type=str, default=None, help='Optional path to save metrics JSON')
	parser.add_argument('--seed', type=int, default=None, help='Seed override for deterministic sampling')
	parser.add_argument('--ddp', action='store_true', help='Force DistributedDataParallel evaluation (otherwise auto-detect)')
	parsed = parser.parse_args()

	predictor_path = parsed.predictor_path or config.finetuned_predictor_path
	tokenizer_path = parsed.tokenizer_path or config.finetuned_tokenizer_path
	if parsed.device:
		device = torch.device(parsed.device)
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	max_samples = parsed.max_samples or config.n_fast_test_iter
	batch_size = parsed.batch_size or config.batch_size

	args = FastTestArgs(
		predictor_path=predictor_path,
		tokenizer_path=tokenizer_path,
		device=device,
		max_samples=max_samples,
		batch_size=batch_size,
		num_workers=parsed.num_workers,
		log_interval=parsed.log_interval,
		output_json=parsed.output_json,
	)
	parsed.fast_test_args = args
	return args, parsed


def main():
	config = Config()
	args, parsed = parse_cli(config)
	env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
	use_ddp = parsed.ddp or env_world_size > 1
	rank = 0
	world_size = 1
	local_rank = 0

	try:
		if use_ddp:
			rank, world_size, local_rank = setup_ddp()
			args.device = torch.device(f'cuda:{local_rank}')
			env_info = f"DDP rank {rank}/{world_size} on cuda:{local_rank}"
		else:
			env_info = f"Single-process on {args.device}"

		seed = parsed.seed if parsed.seed is not None else config.seed
		set_seed(seed, rank)

		if rank == 0:
			print("Fast test configuration:")
			print(json.dumps({
				'predictor_path': args.predictor_path,
				'tokenizer_path': args.tokenizer_path,
				'device': str(args.device),
				'max_samples': args.max_samples,
				'batch_size': args.batch_size,
				'num_workers': args.num_workers,
				'log_interval': args.log_interval,
				'ddp_enabled': use_ddp,
				'environment': env_info,
			}, indent=4))

		run_fast_test(config, args, rank=rank, world_size=world_size, use_ddp=use_ddp)
	finally:
		if use_ddp:
			cleanup_ddp()


if __name__ == '__main__':
	main()
