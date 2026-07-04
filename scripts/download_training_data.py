"""Download short single-domain protein backbones to train the generator.

Phase 2's generator needs a set of small monomer backbones. Two sources:

* ``rcsb`` (default) — query the RCSB Search API for single-chain protein
  structures whose length falls in ``[min_len, max_len]`` (X-ray, good
  resolution), giving a sizeable, fresh training set automatically.
* ``list`` — a curated fallback list of classic small folds (used automatically
  if the RCSB query fails or returns nothing).

The default ``--min-len`` matches the generator's crop length (64), so every
downloaded chain is actually usable — no silent waste. Extracted chains are
written to ``data/processed/train`` as Cα-bearing PDBs that
``PDBBackboneDataset`` reads.

Usage:
    python scripts/download_training_data.py                 # RCSB, ~80 structures
    python scripts/download_training_data.py --count 150
    python scripts/download_training_data.py --source list   # curated fallback
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

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# Curated fallback: classic small single-domain proteins (40–128 residues).
DEFAULT_PDB_IDS = [
    "1PGB", "2GB1", "1UBQ", "1SHG", "1BDD", "1ENH", "1MJC", "1CSP", "5PTI",
    "1FKB", "1SRL", "1TEN", "1WIT", "1HZ6", "1URN", "1AAP", "1FYN", "1G6P",
    "1NYF", "1PRB", "1RIS", "1TUD", "2ACY", "2PTL", "1BEO", "1CTF",
]


def build_rcsb_query(min_len: int, max_len: int, count: int) -> dict:
    """Build the RCSB Search API query for small single-chain protein entries."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match", "value": "Protein"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length",
                    "operator": "range",
                    "value": {"from": min_len, "to": max_len,
                              "include_lower": True, "include_upper": True}}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                    "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal", "value": 2.5}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": count},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def query_rcsb(min_len: int, max_len: int, count: int, timeout: int = 60) -> list[str]:
    """Return PDB ids from the RCSB Search API (empty list on failure)."""
    import requests

    try:
        resp = requests.post(
            RCSB_SEARCH_URL, json=build_rcsb_query(min_len, max_len, count), timeout=timeout
        )
        resp.raise_for_status()
        results = resp.json().get("result_set", [])
        return [r["identifier"] for r in results]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("RCSB search failed (%s); falling back to curated list.", exc)
        return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download short protein backbones for training.")
    p.add_argument("--source", choices=["rcsb", "list"], default="rcsb")
    p.add_argument("--pdb-list", default=None, help="File with one PDB id per line (source=list).")
    p.add_argument("--count", type=int, default=80, help="Max structures to fetch (source=rcsb).")
    p.add_argument("--min-len", type=int, default=64, help="Matches the generator crop length.")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "train"))
    p.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "train"))
    return p.parse_args()


def load_ids(args) -> list[str]:
    if args.source == "list":
        if args.pdb_list:
            return [ln.strip().upper() for ln in Path(args.pdb_list).read_text().splitlines() if ln.strip()]
        return list(DEFAULT_PDB_IDS)
    ids = query_rcsb(args.min_len, args.max_len, args.count)
    if not ids:
        return list(DEFAULT_PDB_IDS)
    LOGGER.info("RCSB returned %d candidate structures.", len(ids))
    return ids


def main() -> int:
    from scripts.download_data import download_structure, extract_chain  # reuse Phase 0 helpers

    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = load_ids(args)
    LOGGER.info("Fetching %d structures (crop-usable length %d–%d) ...", len(ids), args.min_len, args.max_len)
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
        n_ca = sum(
            1 for ln in out.read_text().splitlines()
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA"
        )
        if args.min_len <= n_ca <= args.max_len:
            kept += 1
        else:
            out.unlink(missing_ok=True)
            skipped += 1

    LOGGER.info("Done. Kept %d, skipped %d -> %s", kept, skipped, out_dir)
    LOGGER.info(
        "Train:  python scripts/train_generator.py --data-dir %s --length %d --epochs 150",
        out_dir, args.min_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
