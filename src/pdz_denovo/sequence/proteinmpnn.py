"""Inverse folding: assign sequences to generated Cα backbones.

The **Design → Build** hand-off of the DBTL loop. Phase 2 produces Cα backbones;
this module turns each backbone into candidate amino-acid sequences using
**ProteinMPNN** (Dauparas et al., 2022) — the field-standard inverse-folding
model — specifically its **Cα-only** variant, which matches the Cα traces our
flow model generates. Designed sequences then have the PDZ class-I motif grafted
on (:mod:`pdz_denovo.sequence.motif`) and are returned as
:class:`~pdz_denovo.oracle.types.Candidate` objects ready for the oracle stack.

Two designers share one interface:

* :class:`ProteinMPNNDesigner` — wraps the official ``protein_mpnn_run.py`` via a
  subprocess (the stable CLI, so we never depend on ProteinMPNN's internal API).
  Clone the MIT-licensed repo first with ``scripts/download_proteinmpnn.py``.
* :class:`FallbackDesigner` — a dependency-free stub that emits random sequences
  with the motif grafted. It exists so the full DBTL loop (Phase 5) and the test
  suite can run *before* ProteinMPNN is installed. It is **not** a real designer;
  it makes no use of structure.
"""
from __future__ import annotations

import logging
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from pdz_denovo.oracle.types import AA_ALPHABET, Candidate
from pdz_denovo.sequence.motif import graft_motif, motif_satisfied

LOGGER = logging.getLogger("pdz_denovo")


def pdb_ca_length(pdb_path: str | Path) -> int:
    """Number of Cα atoms in a PDB file (i.e. the backbone length)."""
    n = 0
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            n += 1
    return n


def parse_mpnn_fasta(text: str, skip_first: bool = True) -> list[tuple[str, str]]:
    """Parse a ProteinMPNN output FASTA into ``(header, sequence)`` records.

    ProteinMPNN writes the input (native) sequence as the first record followed
    by the sampled designs; ``skip_first`` drops that native entry.
    """
    records: list[tuple[str, str]] = []
    header, seq_parts = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header, seq_parts = line[1:].strip(), []
        elif line.strip():
            seq_parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_parts)))
    if skip_first and records:
        records = records[1:]
    return records


class ProteinMPNNDesigner:
    """Structure-conditioned sequence design via the ProteinMPNN CLI."""

    def __init__(
        self,
        repo_dir: str | Path,
        ca_only: bool = True,
        python_exe: str | None = None,
        model_name: str = "v_48_020",
        weights_dir: str | Path | None = None,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.run_py = self.repo_dir / "protein_mpnn_run.py"
        self.ca_only = ca_only
        # Default to the *current* interpreter so ProteinMPNN runs in the same
        # (venv) environment as us — a bare "python" can resolve to a different
        # system Python that lacks numpy/torch.
        self.python_exe = python_exe or sys.executable
        self.model_name = model_name
        # Pass the weights folder explicitly. ProteinMPNN's own auto-detection
        # uses rfind("/"), which is broken on Windows (backslash paths) and
        # produces a malformed weights path; supplying it directly avoids that.
        if weights_dir is not None:
            self.weights_dir = Path(weights_dir).resolve()
        else:
            sub = "ca_model_weights" if ca_only else "vanilla_model_weights"
            self.weights_dir = self.repo_dir / sub

    def _build_command(
        self,
        pdb_path: str | Path,
        out_dir: str | Path,
        n_seqs: int,
        temperature: float,
        seed: int,
        batch_size: int,
    ) -> list[str]:
        cmd = [
            self.python_exe,
            str(self.run_py),
            # Bare filename only: ProteinMPNN derives the output name by splitting
            # on "/", which mangles Windows paths. We run with cwd = the PDB's
            # folder (see design()), so the basename resolves correctly and the
            # output name stays clean.
            "--pdb_path",
            Path(pdb_path).name,
            "--out_folder",
            str(Path(out_dir).resolve()),
            "--path_to_model_weights",
            str(self.weights_dir),
            "--model_name",
            self.model_name,
            "--num_seq_per_target",
            str(n_seqs),
            "--sampling_temp",
            str(temperature),
            "--seed",
            str(seed),
            "--batch_size",
            str(batch_size),
        ]
        if self.ca_only:
            cmd.append("--ca_only")
        return cmd

    def design(
        self,
        pdb_path: str | Path,
        n_seqs: int = 8,
        temperature: float = 0.1,
        seed: int = 37,
        batch_size: int = 1,
        enforce_motif: bool = True,
        backbone_id: str | None = None,
    ) -> list[Candidate]:
        """Design ``n_seqs`` sequences for one backbone PDB.

        Returns a list of :class:`Candidate` (origin ``"proteinmpnn"``).
        """
        pdb_path = Path(pdb_path).resolve()
        if not self.run_py.exists():
            raise FileNotFoundError(
                f"ProteinMPNN not found at {self.run_py}. Clone it first:\n"
                f"    python scripts/download_proteinmpnn.py"
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_abs = Path(tmp).resolve()
            cmd = self._build_command(
                pdb_path, tmp_abs, n_seqs, temperature, seed, batch_size
            )
            LOGGER.info("Running ProteinMPNN: %s", " ".join(cmd))
            try:
                # cwd = PDB folder so the bare --pdb_path basename resolves.
                subprocess.run(
                    cmd, check=True, capture_output=True, text=True,
                    cwd=str(pdb_path.parent),
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"ProteinMPNN failed (exit {exc.returncode}).\n"
                    f"--- stderr ---\n{exc.stderr}\n--- stdout ---\n{exc.stdout}"
                ) from exc
            fasta = tmp_abs / "seqs" / f"{pdb_path.stem}.fa"
            records = parse_mpnn_fasta(fasta.read_text())

        return _records_to_candidates(
            records, origin="proteinmpnn", enforce_motif=enforce_motif,
            backbone_id=backbone_id or pdb_path.stem,
        )


class FallbackDesigner:
    """Structure-agnostic random designer (stub for pre-ProteinMPNN runs).

    Emits random sequences of the backbone length with the PDZ motif grafted so
    downstream stages have valid inputs. Clearly **not** a substitute for real
    inverse folding — its designs ignore the backbone entirely.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def design(
        self,
        pdb_path: str | Path | None = None,
        length: int | None = None,
        n_seqs: int = 8,
        enforce_motif: bool = True,
        backbone_id: str | None = None,
        **_ignore,
    ) -> list[Candidate]:
        if length is None:
            if pdb_path is None:
                raise ValueError("Provide either 'length' or 'pdb_path'.")
            length = pdb_ca_length(pdb_path)
        records = []
        for i in range(n_seqs):
            seq = "".join(self._rng.choice(AA_ALPHABET) for _ in range(length))
            records.append((f"fallback,sample={i}", seq))
        return _records_to_candidates(
            records, origin="fallback", enforce_motif=enforce_motif,
            backbone_id=backbone_id or (Path(pdb_path).stem if pdb_path else "random"),
        )


def _records_to_candidates(
    records: list[tuple[str, str]],
    origin: str,
    enforce_motif: bool,
    backbone_id: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for header, seq in records:
        seq = seq.upper()
        if enforce_motif:
            seq = graft_motif(seq)
        candidates.append(
            Candidate(
                sequence=seq,
                origin=origin,
                metadata={
                    "backbone_id": backbone_id,
                    "mpnn_header": header,
                    "motif_satisfied": motif_satisfied(seq),
                },
            )
        )
    return candidates
