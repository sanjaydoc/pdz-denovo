"""Download short single-domain protein backbones to train the generator.

Phase 2's generator needs a set of small monomer backbones. This fetches a
curated list of short, well-folded single-domain proteins from RCSB, extracts
the target chain, keeps those in the length window, and writes them to
``data/processed/train`` as Cα-bearing PDBs that ``PDBBackboneDataset`` reads.

The default list is a set of classic small folds (villin, protein G/A, SH3,
ubiquitin, cold-shock, homeodomain, …). Extend it with ``--pdb-list file.txt``
(one PDB id per line) for a larger training set.

Usage:
    python scripts/download_training_data.py
    python scripts/download_training_data.py --min-len 40 --max-len 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `scripts.download_data` resolves

from pdz_denovo.utils.common import setup_logging  # noqa: E402

LOGGER = setup_logging()

# Classic small single-domain proteins (40–128 residues, mostly monomeric).
DEFAULT_PDB_IDS = [
    "1VII", "1PGB", "2GB1", "1UBQ", "1SHG", "1BDD", "1ENH", "1MJC", "1CSP",
    "5PTI", "1FKB", "2CI2", "1SRL", "1PIN", "1TEN", "1FNF", "1WIT", "1HZ6",
    "1AYE", "1DIV", "1URN", "1AAP", "1FYN", "1G6P", "1NYF", "1PRB", "1RIS",
    "1STN", "1TUD", "2ACY", "2CDS", "2PTL", "1AKI", "1BEO", "1COA", "1CTF",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download short protein backbones for training.")
    p.add_argument("--pdb-list", default=None, help="File with one PDB id per line.")
    p.add_argument("--min-len", type=int, default=40)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "train"))
    p.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "train"))
    return p.parse_args()


def load_ids(args) -> list[str]:
    if args.pdb_list:
        ids = [ln.strip().upper() for ln in Path(args.pdb_list).read_text().splitlines() if ln.strip()]
        return ids
    return list(DEFAULT_PDB_IDS)


def main() -> int:
    from scripts.download_data import download_structure, extract_chain  # reuse Phase 0 helpers

    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = load_ids(args)
    LOGGER.info("Fetching %d structures ...", len(ids))
    kept, skipped = 0, 0
    for pdb_id in ids:
        try:
            pdb_path = download_structure(pdb_id, "pdb", raw_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s (download failed: %s)", pdb_id, exc)
            skipped += 1
            continue
        out = extract_chain(pdb_path, "A", out_dir)
        if out is None:
            skipped += 1
            continue
        # Length filter on the extracted chain.
        n_ca = sum(
            1 for ln in out.read_text().splitlines()
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA"
        )
        if args.min_len <= n_ca <= args.max_len:
            kept += 1
            LOGGER.info("kept %s (%d residues)", pdb_id, n_ca)
        else:
            out.unlink(missing_ok=True)
            skipped += 1
            LOGGER.info("dropped %s (%d residues out of [%d,%d])", pdb_id, n_ca, args.min_len, args.max_len)

    LOGGER.info("Done. Kept %d, skipped %d -> %s", kept, skipped, out_dir)
    LOGGER.info("Train the generator with:  python scripts/train_generator.py --data-dir %s --epochs 150", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
