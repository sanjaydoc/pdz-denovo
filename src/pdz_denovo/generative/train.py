"""Training loop for the SE(3) flow-matching backbone generator.

Tuned for a 6 GB GPU: fp16 autocast (``mixed_precision``), optional gradient
checkpointing, small batches, and gradient clipping. Works CPU-only too (just
slower). Logs to MLflow when available and always writes a JSON metrics trail.

Usage (see also ``scripts/train_generator.py``)::

    python -m pdz_denovo.generative.train --epochs 50 --data-dir data/processed
    python -m pdz_denovo.generative.train --synthetic --epochs 5   # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pdz_denovo.utils.common import get_project_root, resolve_device, set_seed, setup_logging

LOGGER = logging.getLogger("pdz_denovo")


def train(
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    length: int = 64,
    data_dir: str | None = None,
    synthetic: bool = False,
    device: str = "auto",
    out_dir: str | Path = "outputs/generator",
    grad_clip: float = 1.0,
    mixed_precision: bool = True,
    seed: int = 42,
    flow_cfg: dict | None = None,
    log_every: int = 20,
) -> Path:
    """Train the generator and return the path to the saved checkpoint."""
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from pdz_denovo.generative.dataset import SyntheticBackboneDataset, build_dataset
    from pdz_denovo.generative.flow import build_flow_model

    setup_logging()
    set_seed(seed)
    device = resolve_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- config --------------------------------------------------------------
    if flow_cfg is None:
        cfg_path = get_project_root() / "configs" / "model" / "flow.yaml"
        flow_cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else OmegaConf.create({})
    else:
        flow_cfg = OmegaConf.create(flow_cfg)
    # Fill defaults so the model builds even from a partial config.
    defaults = {
        "hidden_dim": 128,
        "n_layers": 5,
        "edge_dim": 32,
        "n_neighbors": 16,
        "max_residues": max(80, length),
        "time_embed_dim": 32,
        "grad_checkpoint": True,
    }
    for key, val in defaults.items():
        if key not in flow_cfg:
            flow_cfg[key] = val

    # --- data ----------------------------------------------------------------
    if synthetic or data_dir is None:
        dataset = SyntheticBackboneDataset(n=512, length=length)
    else:
        dataset = build_dataset(length=length, pdb_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    LOGGER.info("Training on %d samples (%s).", len(dataset), type(dataset).__name__)

    # --- model ---------------------------------------------------------------
    model = build_flow_model(flow_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = mixed_precision and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("Model: %.2fM params on %s (amp=%s).", n_params / 1e6, device, use_amp)

    # --- optional MLflow -----------------------------------------------------
    mlflow = None
    try:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_experiment("pdz-denovo-generator")
        mlflow.start_run()
        mlflow.log_params(
            {"epochs": epochs, "batch_size": batch_size, "lr": lr, "length": length}
        )
    except Exception:  # noqa: BLE001 - tracking is optional
        LOGGER.info("MLflow not available; logging to JSON only.")

    history = []
    for epoch in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for coords in loader:
            coords = coords.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = model.loss(coords)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n_batches += 1
        epoch_loss = running / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if epoch % max(log_every, 1) == 0 or epoch == epochs - 1:
            LOGGER.info("epoch %3d | loss %.4f", epoch, epoch_loss)
        if mlflow is not None:
            mlflow.log_metric("loss", epoch_loss, step=epoch)

    ckpt = out_dir / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "flow_cfg": OmegaConf.to_container(flow_cfg)},
        ckpt,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    LOGGER.info("Saved checkpoint -> %s", ckpt)
    if mlflow is not None:
        mlflow.log_artifact(str(ckpt))
        mlflow.end_run()
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the SE(3) flow-matching generator.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--data-dir", default=None, help="Dir of PDB files; omit for synthetic.")
    p.add_argument("--synthetic", action="store_true", help="Force synthetic backbones.")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="outputs/generator")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        length=args.length,
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        device=args.device,
        out_dir=args.out_dir,
        mixed_precision=not args.no_amp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
