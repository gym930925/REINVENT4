"""Programmatic NaviDiv demo — no Streamlit required.

Loads sample_molecules.csv (bundled in this directory), runs all diversity
metrics, prints a summary table, and writes outputs to ./demo_output/.

Usage:
    conda activate NaviDiv
    python run_demo.py

Requirements: navidiv installed (pip install -e /path/to/NaviDiv)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
SAMPLE_CSV = SCRIPT_DIR / "sample_molecules.csv"
OUTPUT_DIR = SCRIPT_DIR / "demo_output"

# ── helpers ──────────────────────────────────────────────────────────────────

def _require_navidiv() -> None:
    try:
        import navidiv  # noqa: F401
    except ImportError:
        sys.exit(
            "NaviDiv is not installed.\n"
            "  pip install -e /path/to/NaviDiv\n"
            "or activate the correct conda environment."
        )


def _load_smiles(csv_path: Path) -> tuple[list[str], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    col = next(
        (c for c in df.columns if c.lower() in {"smiles", "substructure"}),
        None,
    )
    if col is None:
        sys.exit(f"No SMILES column found in {csv_path}")
    return df[col].dropna().tolist(), df


# ── main demo ────────────────────────────────────────────────────────────────

def run_global_diversity(smiles: list[str]) -> None:
    from navidiv.diversity.diversity import diversity_all

    modes = ["Richness", "IntDiv", "BM", "FG", "RS"]
    print("\n── Global diversity metrics ────────────────────────────────")
    print(f"  {'Mode':<12}  {'Score':>8}")
    print(f"  {'-'*12}  {'-'*8}")
    for mode in modes:
        try:
            score = diversity_all(smiles=smiles, mode=mode)
            print(f"  {mode:<12}  {score:>8.4f}")
        except Exception as exc:
            print(f"  {mode:<12}  ERROR: {exc}")


def run_scorers(smiles: list[str], df: pd.DataFrame, output_dir: Path) -> None:
    from navidiv.utils import initialize_scorer

    scorer_configs = [
        {"scorer_name": "Scaffold", "scaffold_type": "basic_wire_frame",
         "min_count_fragments": 1},
        {"scorer_name": "Ngram", "ngram_size": 10, "min_count_fragments": 3},
        {"scorer_name": "Fragments", "transformation_mode": "none",
         "min_count_fragments": 2},
        {"scorer_name": "RingScorer"},
        {"scorer_name": "FGscorer"},
    ]

    step_col = next(
        (c for c in df.columns if c.lower() == "step"), None
    )
    score_col = next(
        (c for c in df.columns if c.lower() == "score"), None
    )

    print("\n── Per-scorer analysis ─────────────────────────────────────")
    for props in scorer_configs:
        name = props["scorer_name"]
        scorer_output = output_dir / name
        scorer_output.mkdir(parents=True, exist_ok=True)
        props["output_path"] = str(scorer_output)

        try:
            scorer = initialize_scorer(props)
            extra = pd.DataFrame()
            if step_col:
                extra["step"] = df[step_col].values
            if score_col:
                extra["Score"] = df[score_col].values

            result = scorer.get_score(
                smiles_list=smiles,
                scores=df[score_col].values if score_col else None,
                additional_columns_df=extra if not extra.empty else None,
            )
            print(f"  {name}: {result}")
        except Exception as exc:
            print(f"  {name}: ERROR — {exc}")


def run_tsne(csv_path: Path, output_dir: Path) -> Path | None:
    try:
        from navidiv.get_tsne import compute_tsne
    except ImportError:
        print("  t-SNE: skipped (get_tsne not importable as a module)")
        return None

    out_path = output_dir / (csv_path.stem + "_TSNE.csv")
    try:
        compute_tsne(df_path=str(csv_path), step=20, output_path=str(out_path))
        print(f"  t-SNE written to {out_path.relative_to(SCRIPT_DIR)}")
        return out_path
    except Exception as exc:
        print(f"  t-SNE: ERROR — {exc}")
        return None


def main() -> None:
    _require_navidiv()

    if not SAMPLE_CSV.exists():
        sys.exit(f"Sample CSV not found: {SAMPLE_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SAMPLE_CSV.name} ...")
    smiles, df = _load_smiles(SAMPLE_CSV)
    print(f"  {len(smiles)} molecules loaded")

    run_global_diversity(smiles)
    run_scorers(smiles, df, OUTPUT_DIR)

    print("\n── t-SNE projection ────────────────────────────────────────")
    run_tsne(SAMPLE_CSV, OUTPUT_DIR)

    print(f"\nDone. Outputs written to {OUTPUT_DIR.relative_to(Path.cwd(), 'error') if False else OUTPUT_DIR}")
    print("\nTo explore results interactively:")
    print("  conda activate NaviDiv")
    print("  streamlit run /path/to/NaviDiv/app.py")
    print(f"  # then load: {OUTPUT_DIR}/<scorer>/...")


if __name__ == "__main__":
    main()
