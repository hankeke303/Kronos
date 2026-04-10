import os
import sys
import json
import time
from time import gmtime, strftime
import argparse
import csv
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import comet_ml

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)


# revised by ZMJ. 2026-1-17
def compute_ccc_loss(pred, target):
    """
    Compute Concordance Correlation Coefficient (CCC) loss.
    Only computed on close_return column (index 9).
    
    Args:
        pred (torch.Tensor): Predicted values, shape [batch_size, seq_len, d_in]
        target (torch.Tensor): Target values, shape [batch_size, seq_len, d_in]
    
    Returns:
        torch.Tensor: CCC loss (1 - CCC), scalar
    """
    # revised by ZMJ. 2026-1-17: Only compute CCC on close_return column (index 9)
    feature_idx = 9  # close_return column index
    # Keep statistics in fp32 for stability under bf16 training.
    pred_feature = pred[:, :, feature_idx].reshape(-1).float()  # [N]
    target_feature = target[:, :, feature_idx].reshape(-1).float()  # [N]
    
    # Compute means and standard deviations
    pred_mean = pred_feature.mean()
    target_mean = target_feature.mean()
    pred_std = pred_feature.std()
    target_std = target_feature.std()
    
    # Compute Pearson correlation coefficient
    pred_centered = pred_feature - pred_mean  # [N]
    target_centered = target_feature - target_mean  # [N]
    numerator = (pred_centered * target_centered).mean()
    denominator = pred_std * target_std + 1e-8
    rho = numerator / denominator
    
    # Compute CCC
    ccc_numerator = 2 * rho * pred_std * target_std
    ccc_denominator = pred_std ** 2 + target_std ** 2 + (pred_mean - target_mean) ** 2 + 1e-8
    ccc = ccc_numerator / ccc_denominator
    
    # Convert to loss (1 - CCC)
    ccc_loss = 1.0 - ccc
    
    return ccc_loss


def decode_topk_weighted(tokenizer, s1_logits, s2_logits, top_k):
    """
    Decode weighted top-k token pairs in a vectorized way.
    This keeps the same math as full-softmax->topk->renorm, but runs faster:
    - topk on logits (equivalent ranking to probs)
    - renormalize only inside top-k set
    - single batched decode instead of K small decode calls
    """
    top_k_eff = min(top_k, s1_logits.shape[-1], s2_logits.shape[-1])

    s1_topk_logits, s1_topk_indices = torch.topk(s1_logits, k=top_k_eff, dim=-1)
    s2_topk_logits, s2_topk_indices = torch.topk(s2_logits, k=top_k_eff, dim=-1)

    s1_topk_probs_norm = F.softmax(s1_topk_logits, dim=-1)
    s2_topk_probs_norm = F.softmax(s2_topk_logits, dim=-1)

    s1_tokens_all = s1_topk_indices.permute(2, 0, 1).reshape(-1, s1_logits.shape[1])
    s2_tokens_all = s2_topk_indices.permute(2, 0, 1).reshape(-1, s2_logits.shape[1])
    with torch.no_grad():
        decoded_all = tokenizer.decode([s1_tokens_all, s2_tokens_all], half=True)
    pred_values_stack = decoded_all.reshape(
        top_k_eff, s1_logits.shape[0], s1_logits.shape[1], decoded_all.shape[-1]
    )

    weights_stack = s1_topk_probs_norm.permute(2, 0, 1) * s2_topk_probs_norm.permute(2, 0, 1)
    weights_stack = weights_stack / (weights_stack.sum(dim=0, keepdim=True) + 1e-8)
    weights_stack = weights_stack.unsqueeze(-1)

    return (pred_values_stack * weights_stack).sum(dim=0)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    test_dataset = QlibDataset('test')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], sampler=test_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, test_loader, train_dataset, valid_dataset, test_dataset


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    train_loader, val_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=config['predictor_learning_rate'],
    #     steps_per_epoch=len(train_loader), epochs=config['epochs'],
    #     pct_start=0.03, div_factor=10
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience_limit = config.get('early_stop_patience', 999999)
    dt_result = {}
    batch_idx_global = 0
    epochs_ran = 0

    metrics_csv_path = os.path.join(save_dir, "metrics.csv")
    # top_k = 10
    top_k = 3
    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        last_log_time = epoch_start_time
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)
        test_dataset.set_epoch_seed(0)
        
        num_iters = len(train_loader)
        epoch_train_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_ccc_loss_sum = 0.0
        epoch_combine_loss_sum = 0.0

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.squeeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
            batch_x_stamp = batch_x_stamp.squeeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            ce_loss, s1_loss, s2_loss = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # revised by ZMJ. 2026-1-17
            # Get top-k predicted tokens from logits using Straight-Through Estimator (STE)
            # Forward: use topk indices (not differentiable), Backward: gradient flows through softmax probabilities
            s1_logits, s2_logits = logits[0], logits[1]  # [batch_size, seq_len, vocab_size]
            batch_size, seq_len = s1_logits.shape[0], s1_logits.shape[1]
            half_start = seq_len // 2
            s1_logits = s1_logits[:, half_start:, :]
            s2_logits = s2_logits[:, half_start:, :]
            
            # Use top-k decode with STE weights (vectorized for speed/memory)
            pred_values = decode_topk_weighted(tokenizer, s1_logits, s2_logits, top_k)
            
            # Get true values corresponding to token_out positions (batch_x[:, 1:])
            true_values = batch_x[:, 1 + half_start:, :]  # [batch_size, seq_len/2, d_in]
            
            # Compute CCC loss
            ccc_loss = compute_ccc_loss(pred_values, true_values)
            
            # Combine original loss and CCC loss (average)
            combine_loss = (ccc_loss + ce_loss) / 2.0

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            combine_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging (Master Process Only)
            batch_ce_loss_value = ce_loss.item()
            batch_ccc_loss_value = ccc_loss.item()
            batch_loss_value = combine_loss.item()
            epoch_train_loss_sum += batch_loss_value
            epoch_ce_loss_sum += batch_ce_loss_value
            epoch_ccc_loss_sum += batch_ccc_loss_value
            epoch_combine_loss_sum += batch_loss_value
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                now = time.time()
                interval_steps = config['log_interval']
                interval_time = now - last_log_time
                step_time = interval_time / max(interval_steps, 1)
                epoch_elapsed = now - epoch_start_time
                last_log_time = now
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Combine Loss: {batch_loss_value:.4f}, CE Loss: {batch_ce_loss_value:.4f}, CCC Loss: {batch_ccc_loss_value:.4f}, "
                    f"Step Time: {step_time:.4f}s, Interval Time: {format_time(interval_time)}, Epoch Elapsed: {format_time(epoch_elapsed)}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_ce_loss_batch', batch_ce_loss_value, step=batch_idx_global)
                logger.log_metric('train_ccc_loss_batch', batch_ccc_loss_value, step=batch_idx_global)
                logger.log_metric('train_combine_loss_batch', batch_loss_value, step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_ce_loss_rank = 0.0
        tot_val_ccc_loss_rank = 0.0
        tot_val_combine_loss_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.squeeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                ce_loss, _, _ = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                # Compute CCC loss for validation
                s1_logits, s2_logits = logits[0], logits[1]
                seq_len = s1_logits.shape[1]
                half_start = seq_len // 2
                s1_logits = s1_logits[:, half_start:, :]
                s2_logits = s2_logits[:, half_start:, :]
                pred_values = decode_topk_weighted(tokenizer, s1_logits, s2_logits, top_k)
                true_values = batch_x[:, 1 + half_start:, :]
                ccc_loss = compute_ccc_loss(pred_values, true_values)
                combine_loss = (ccc_loss + ce_loss) / 2.0

                tot_val_ce_loss_rank += ce_loss.item()
                tot_val_ccc_loss_rank += ccc_loss.item()
                tot_val_combine_loss_rank += combine_loss.item()
                val_batches_processed_rank += 1

        # Reduce validation metrics
        val_loss_sum_tensor = torch.tensor([tot_val_ce_loss_rank, tot_val_ccc_loss_rank, tot_val_combine_loss_rank], device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_ce_loss = val_loss_sum_tensor[0].item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_ccc_loss = val_loss_sum_tensor[1].item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_combine_loss = val_loss_sum_tensor[2].item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_loss = avg_val_combine_loss  # For backward compatibility

        # --- Test Loop (no console output) ---
        tot_test_ce_loss_rank = 0.0
        tot_test_ccc_loss_rank = 0.0
        tot_test_combine_loss_rank = 0.0
        test_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in test_loader:
                batch_x = batch_x.squeeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                ce_loss, _, _ = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                # Compute CCC loss for test
                s1_logits, s2_logits = logits[0], logits[1]
                seq_len = s1_logits.shape[1]
                half_start = seq_len // 2
                s1_logits = s1_logits[:, half_start:, :]
                s2_logits = s2_logits[:, half_start:, :]
                pred_values = decode_topk_weighted(tokenizer, s1_logits, s2_logits, top_k)
                true_values = batch_x[:, 1 + half_start:, :]
                ccc_loss = compute_ccc_loss(pred_values, true_values)
                combine_loss = (ccc_loss + ce_loss) / 2.0

                tot_test_ce_loss_rank += ce_loss.item()
                tot_test_ccc_loss_rank += ccc_loss.item()
                tot_test_combine_loss_rank += combine_loss.item()
                test_batches_processed_rank += 1

        test_loss_sum_tensor = torch.tensor([tot_test_ce_loss_rank, tot_test_ccc_loss_rank, tot_test_combine_loss_rank], device=device)
        test_batches_tensor = torch.tensor(test_batches_processed_rank, device=device)
        dist.all_reduce(test_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_batches_tensor, op=dist.ReduceOp.SUM)
        avg_test_ce_loss = test_loss_sum_tensor[0].item() / test_batches_tensor.item() if test_batches_tensor.item() > 0 else 0
        avg_test_ccc_loss = test_loss_sum_tensor[1].item() / test_batches_tensor.item() if test_batches_tensor.item() > 0 else 0
        avg_test_combine_loss = test_loss_sum_tensor[2].item() / test_batches_tensor.item() if test_batches_tensor.item() > 0 else 0
        avg_test_loss = avg_test_combine_loss  # For backward compatibility

        # --- Aggregate train loss across ranks ---
        train_loss_sum_tensor = torch.tensor([epoch_ce_loss_sum, epoch_ccc_loss_sum, epoch_combine_loss_sum], device=device)
        dist.all_reduce(train_loss_sum_tensor, op=dist.ReduceOp.SUM)
        avg_ce_loss = train_loss_sum_tensor[0].item() / (num_iters * world_size) if num_iters > 0 else 0
        avg_ccc_loss = train_loss_sum_tensor[1].item() / (num_iters * world_size) if num_iters > 0 else 0
        avg_combine_loss = train_loss_sum_tensor[2].item() / (num_iters * world_size) if num_iters > 0 else 0
        avg_train_loss = avg_combine_loss  # For backward compatibility

        current_lr = optimizer.param_groups[0]['lr']

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Train - CE Loss: {avg_ce_loss:.4f}, CCC Loss: {avg_ccc_loss:.4f}, Combine Loss: {avg_combine_loss:.4f}")
            print(f"Validation - CE Loss: {avg_val_ce_loss:.4f}, CCC Loss: {avg_val_ccc_loss:.4f}, Combine Loss: {avg_val_combine_loss:.4f}")
            print(f"Test - CE Loss: {avg_test_ce_loss:.4f}, CCC Loss: {avg_test_ccc_loss:.4f}, Combine Loss: {avg_test_combine_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")

            if logger:
                # Train metrics
                logger.log_metric('train_ce_loss_epoch', avg_ce_loss, epoch=epoch_idx)
                logger.log_metric('train_ccc_loss_epoch', avg_ccc_loss, epoch=epoch_idx)
                logger.log_metric('train_combine_loss_epoch', avg_combine_loss, epoch=epoch_idx)
                # Validation metrics
                logger.log_metric('val_ce_loss_epoch', avg_val_ce_loss, epoch=epoch_idx)
                logger.log_metric('val_ccc_loss_epoch', avg_val_ccc_loss, epoch=epoch_idx)
                logger.log_metric('val_combine_loss_epoch', avg_val_combine_loss, epoch=epoch_idx)
                # Test metrics
                logger.log_metric('test_ce_loss_epoch', avg_test_ce_loss, epoch=epoch_idx)
                logger.log_metric('test_ccc_loss_epoch', avg_test_ccc_loss, epoch=epoch_idx)
                logger.log_metric('test_combine_loss_epoch', avg_test_combine_loss, epoch=epoch_idx)
                # logger.log_metric('predictor_learning_rate_epoch', current_lr, epoch=epoch_idx)

            if avg_val_combine_loss < best_val_loss:
                best_val_loss = avg_val_combine_loss
                best_epoch = epoch_idx + 1
                patience_counter = 0
                save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

            else:
                patience_counter += 1

            # Save metrics to CSV after each epoch (append mode)
            header = ['epoch', 'learning_rate', 'train_ce_loss', 'train_ccc_loss', 'train_combine_loss', 'val_ce_loss', 'val_ccc_loss', 'val_combine_loss', 'test_ce_loss', 'test_ccc_loss', 'test_combine_loss']
            file_exists = os.path.exists(metrics_csv_path)
            with open(metrics_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow([epoch_idx + 1, current_lr, avg_ce_loss, avg_ccc_loss, avg_combine_loss, avg_val_ce_loss, avg_val_ccc_loss, avg_val_combine_loss, avg_test_ce_loss, avg_test_ccc_loss, avg_test_combine_loss])

            stop_tensor = torch.tensor(int(patience_counter >= patience_limit), device=device)
        else:
            stop_tensor = torch.tensor(0, device=device)

        dist.broadcast(stop_tensor, src=0)
        epochs_ran = epoch_idx + 1
        if stop_tensor.item():
            if rank == 0:
                print(f"Early stopping triggered after {epochs_ran} epochs (patience={patience_limit}).")
            break

        dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    dt_result['best_epoch'] = best_epoch
    dt_result['stopped_early'] = bool(patience_counter >= patience_limit)
    dt_result['epochs_ran'] = epochs_ran
    return dt_result


def _build_predictor_from_scratch(config: dict) -> Kronos:
    return Kronos(
        **config["predictor_model_initialize_params"]
    )


def main(config: dict, mode: str = 'finetune', init: str = 'pretrained', tokenizer_path_override: str | None = None, predictor_path_override: str | None = None, save_folder_name_override: str | None = None):
    """Main function to orchestrate the DDP training process."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_comet']:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    dist.barrier()

    # Model Initialization
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device=device, dtype=torch.bfloat16)
    if init == 'scratch':
        model = _build_predictor_from_scratch(config)
    else:
        load_path = predictor_path_override or config['pretrained_predictor_path']
        model = Kronos.from_pretrained(load_path)
    model.to(device=device, dtype=torch.bfloat16)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Predictor Model Size: {get_model_size(model.module)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py [--mode pretrain|finetune] [--init pretrained|scratch]
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with `torchrun`.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pretrain', 'finetune'], default='finetune')
    parser.add_argument('--init', choices=['pretrained', 'scratch'], default='pretrained')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Override tokenizer path/id')
    parser.add_argument('--predictor_path', type=str, default=None, help='Override predictor pretrained path/id')
    parser.add_argument('--save_folder_name', type=str, default=None, help='Override save folder name')
    args, _ = parser.parse_known_args()

    config_instance = Config()
    main(
        config_instance.__dict__,
        mode=args.mode,
        init=args.init,
        tokenizer_path_override=args.tokenizer_path,
        predictor_path_override=args.predictor_path,
        save_folder_name_override=args.save_folder_name,
    )
