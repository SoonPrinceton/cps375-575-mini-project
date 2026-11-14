"""Train SFT on messages-only JSONL data, save checkpoints, and optionally publish.

Usage:
    python train_sft.py --train_data data/mixtures/balanced_5k.jsonl \
        --output_dir outputs/balanced_5k --checkpoint_name balanced_5k --no_publish
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from dotenv import load_dotenv
from tqdm.auto import tqdm

MODEL = "Qwen/Qwen3.5-9B-Base"  # the only model allowed for the graded submission


def load_data(path, renderer, max_length):
    """Prepare conversations and skip rows without assistant training tokens."""
    data = []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            if not any(message["role"] == "assistant" for message in messages):
                skipped += 1
                continue
            datum = conversation_to_datum(
                messages,
                renderer,
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                reduction="none",  # Preserve token weights for token-weighted loss.
            )
            if not any(weight > 0 for weight in datum.loss_fn_inputs["weights"].tolist()):
                skipped += 1
                continue
            data.append(datum)

    print(f"{path}: {len(data):,} examples prepared; {skipped:,} skipped (no assistant training tokens)")
    if not data:
        raise ValueError(f"No usable examples in {path}")
    return data


def loss_totals(result, batch):
    """Return summed negative log-likelihood and token weight."""
    logprobs = np.concatenate([o["logprobs"].tolist() for o in result.loss_fn_outputs])
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    return float(-np.dot(logprobs, weights)), float(weights.sum())


def evaluate(tc, data, batch_size):
    """Evaluate all validation tokens without gradients or optimizer updates."""
    total_nll = 0.0
    total_weight = 0.0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        result = tc.forward(batch, loss_fn="cross_entropy").result()
        nll, weight = loss_totals(result, batch)
        total_nll += nll
        total_weight += weight
    return total_nll / total_weight, total_weight


def main():
    load_dotenv() 
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", type=Path, required=True, help="Training JSONL path")
    parser.add_argument("--val_data", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=100, help="Validation interval in steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="sft", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    parser.add_argument("--output_dir", type=Path, required=True, help="Local experiment directory")
    args = parser.parse_args()

    for name in ("num_epochs", "batch_size", "rank", "eval_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_length < 2:
        parser.error("--max_length must be at least 2 for next-token prediction")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    for path in (args.train_data, args.val_data):
        if not path.is_file():
            parser.error(f"Dataset not found: {path}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Output directory already exists: {args.output_dir}. Choose a new --output_dir.")

    # Check both sampler and training checkpoints across the user's training runs.
    offset = 0
    try:
        rest_client = tinker.RestClient()
        while True:
            page = rest_client.list_user_checkpoints(limit=100, offset=offset).result()
            for checkpoint in page.checkpoints:
                if checkpoint.tinker_path.rsplit("/", 1)[-1] == args.checkpoint_name:
                    parser.error(
                        f"Checkpoint name already exists: {args.checkpoint_name} "
                        f"({checkpoint.tinker_path}). Choose a new --checkpoint_name."
                    )
            if page.cursor is None:
                break
            offset = page.cursor.offset + page.cursor.limit
            if offset >= page.cursor.total_count:
                break
    except Exception as exc:
        parser.error(f"Unable to check existing checkpoints; training was not started: {exc}")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {args.output_dir}. Choose a new --output_dir.")

    # Setup: use the same tokenizer and renderer as the evaluation scripts.
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    print("Preparing training data...")
    all_data = load_data(args.train_data, renderer, args.max_length)
    print("Preparing validation data...")
    val_data = load_data(args.val_data, renderer, args.max_length)

    # Create training client.
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank, seed=args.seed).result()
    print("  Training client ready")

    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    steps_per_epoch = (len(all_data) + args.batch_size - 1) // args.batch_size
    total_steps = args.num_epochs * steps_per_epoch
    history = {"training": [], "validation": []}
    rng = random.Random(args.seed)
    step = 0
    print(f"\nTraining for {args.num_epochs} epochs, {total_steps} steps "
          f"(batch_size={args.batch_size}, lr={args.lr})...")

    for epoch in range(1, args.num_epochs + 1):
        rng.shuffle(all_data)
        epoch_nll = 0.0
        epoch_weight = 0.0
        progress = tqdm(range(0, len(all_data), args.batch_size), total=steps_per_epoch,
                        desc=f"Epoch {epoch}/{args.num_epochs}", unit="step")
        for start in progress:
            batch = all_data[start:start + args.batch_size]
            fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = tc.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            nll, weight = loss_totals(fwd_bwd_result, batch)
            loss = nll / max(weight, 1)
            epoch_nll += nll
            epoch_weight += weight
            step += 1
            history["training"].append({
                "step": step, "epoch": epoch, "loss": loss, "tokens": weight,
            })
            progress.set_postfix(loss=f"{loss:.4f}", step=f"{step}/{total_steps}")

            # The final step always evaluates, even between regular intervals.
            if step % args.eval_every == 0 or step == total_steps:
                val_loss, val_weight = evaluate(tc, val_data, args.batch_size)
                history["validation"].append({
                    "step": step, "epoch": epoch, "loss": val_loss, "tokens": val_weight,
                })
                tqdm.write(f"  Epoch {epoch}/{args.num_epochs} | Step {step}/{total_steps} "
                           f"| Validation loss: {val_loss:.4f}")
                with (args.output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

        tqdm.write(f"Epoch {epoch}/{args.num_epochs} complete | "
                   f"Mean training loss: {epoch_nll / epoch_weight:.4f}")

    # Save sampler weights for evaluation and full training state for continuation.
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    state = tc.save_state(name=args.checkpoint_name).result()
    print(f"  Sampler checkpoint saved: {checkpoint_path}")
    print(f"  Training state saved: {state.path}")

    info = {
        "checkpoint_path": checkpoint_path,
        "training_state_path": state.path,
        "checkpoint_name": args.checkpoint_name,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "training": {
            "num_epochs": args.num_epochs,
            "num_steps": step,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_length": args.max_length,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
        },
        "published": False,
    }
    info_path = args.output_dir / "checkpoint_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # Publish the sampler checkpoint, preserving the provided evaluation workflow.
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        info["published"] = True
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    print(f"\nCheckpoint info saved to {info_path}")
    print(f"Loss history saved to {args.output_dir / 'loss_history.json'}")
    print("\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()
