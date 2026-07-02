"""Download the PSD-95 PDZ target structure (and seed data) from RCSB PDB.

Phase 0 deliverable. Fetches the reference structure (default: 1BE9, the PSD-95
PDZ3 domain) as both PDB and mmCIF, verifies the download, and extracts the
target chain into data/processed for downstream docking / featurization.

Usage:
    python scripts/download_data.py                # uses configs/config.yaml defaults
    python scripts/download_data.py --pdb-id 1BE9 --chain A
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# Make the package importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdz_denovo.utils.common import setup_logging  # noqa: E402

RCSB_FILE_URL = "https://files.rcsb.org/download/{pdb_id}.{fmt}"
LOGGER = setup_logging()


def download_structure(pdb_id: str, fmt: str, dest_dir: Path) -> Path:
    """Download a single structure file (pdb or cif) from RCSB.

    Args:
        pdb_id: 4-character PDB identifier.
        fmt: "pdb" or "cif".
        dest_dir: directory to write into.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: if the download fails or returns empty content.
    """
    pdb_id = pdb_id.upper()
    url = RCSB_FILE_URL.format(pdb_id=pdb_id, fmt=fmt)
    dest = dest_dir / f"{pdb_id}.{fmt}"
    LOGGER.info("Downloading %s -> %s", url, dest)

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(
            f"Failed to download {pdb_id}.{fmt} (HTTP {resp.status_code})"
        )
    dest.write_bytes(resp.content)
    LOGGER.info("Wrote %d bytes to %s", len(resp.content), dest.name)
    return dest


def extract_chain(pdb_path: Path, chain: str, out_dir: Path) -> Path | None:
    """Extract a single chain into its own PDB file using biotite.

    Falls back gracefully (returns None) if biotite is not yet installed,
    so Phase 0 can run before the full environment is built.
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
    except ImportError:
        LOGGER.warning(
            "biotite not installed yet; skipping chain extraction. "
            "Re-run after environment setup to produce the processed target."
        )
        return None

    pdb_file = pdb.PDBFile.read(str(pdb_path))
    structure = pdb_file.get_structure(model=1)
    mask = (structure.chain_id == chain) & struc.filter_amino_acids(structure)
    chain_struct = structure[mask]
    if chain_struct.array_length() == 0:
        LOGGER.error("Chain %s not found or empty in %s", chain, pdb_path.name)
        return None

    out_path = out_dir / f"{pdb_path.stem}_chain{chain}.pdb"
    out_file = pdb.PDBFile()
    out_file.set_structure(chain_struct)
    out_file.write(str(out_path))
    n_res = struc.get_residue_count(chain_struct)
    LOGGER.info("Extracted chain %s (%d residues) -> %s", chain, n_res, out_path.name)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PDZ target structure from RCSB.")
    p.add_argument("--pdb-id", default="1BE9", help="PDB ID (default: 1BE9, PSD-95 PDZ3)")
    p.add_argument("--chain", default="A", help="Target chain to extract (default: A)")
    p.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw"),
        help="Directory for raw downloads",
    )
    p.add_argument(
        "--processed-dir",
        default=str(ROOT / "data" / "processed"),
        help="Directory for processed outputs",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdb_path = download_structure(args.pdb_id, "pdb", raw_dir)
        download_structure(args.pdb_id, "cif", raw_dir)
    except RuntimeError as exc:
        LOGGER.error("Download failed: %s", exc)
        return 1

    extract_chain(pdb_path, args.chain, processed_dir)
    LOGGER.info("Done. Target '%s' chain '%s' ready.", args.pdb_id, args.chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
