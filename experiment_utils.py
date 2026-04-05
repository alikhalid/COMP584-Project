import math
import time

import torch
from tqdm.auto import tqdm


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [move_batch_to_device(value, device) for value in batch]
        return type(batch)(moved)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def run_model(model, batch):
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (list, tuple)):
        return model(*batch)
    return model(batch)


def default_token_count(batch):
    if isinstance(batch, dict):
        if "target_masks" in batch:
            return batch["target_masks"].sum().item()
        raise ValueError("default_token_count does not know how to count tokens for this batch dict.")
    if isinstance(batch, (list, tuple)) and batch:
        first = batch[0]
        if torch.is_tensor(first):
            return first.numel()
    raise ValueError("default_token_count does not know how to count tokens for this batch.")


def evaluate_model(model, loader, device, max_batches=None, batch_to_device=None, token_count_fn=None):
    model.eval()
    batch_to_device = batch_to_device or move_batch_to_device
    token_count_fn = token_count_fn or default_token_count

    total_loss = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = batch_to_device(batch, device)
            _, loss = run_model(model, batch)

            batch_tokens = token_count_fn(batch)
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def train_model(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    checkpoint_path,
    batch_to_device=None,
    token_count_fn=None,
):
    batch_to_device = batch_to_device or move_batch_to_device
    token_count_fn = token_count_fn or default_token_count

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = {
        "steps": [],
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
    }

    best_val_loss = float("inf")
    step = 0
    running_loss = 0.0
    running_tokens = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)

        for batch in progress:
            if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
                break

            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = run_model(model, batch)
            loss.backward()

            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            batch_tokens = token_count_fn(batch)
            running_loss += loss.item() * batch_tokens
            running_tokens += batch_tokens
            step += 1

            progress.set_postfix(loss=f"{loss.item():.4f}", step=step)

            if step % cfg.eval_every == 0:
                train_loss = running_loss / running_tokens
                train_ppl = math.exp(train_loss)
                val_loss, val_ppl = evaluate_model(
                    model,
                    val_loader,
                    device=device,
                    max_batches=cfg.val_max_batches,
                    batch_to_device=batch_to_device,
                    token_count_fn=token_count_fn,
                )

                history["steps"].append(step)
                history["train_loss"].append(train_loss)
                history["train_ppl"].append(train_ppl)
                history["val_loss"].append(val_loss)
                history["val_ppl"].append(val_ppl)

                print(
                    f"step={step} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
                    f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), checkpoint_path)

        val_loss, val_ppl = evaluate_model(
            model,
            val_loader,
            device=device,
            max_batches=cfg.val_max_batches,
            batch_to_device=batch_to_device,
            token_count_fn=token_count_fn,
        )
        epoch_time = time.time() - epoch_start
        print(f"epoch {epoch + 1} finished in {epoch_time:.1f}s | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
            break

    history["best_val_loss"] = best_val_loss
    return history
