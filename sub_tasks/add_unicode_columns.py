"""
add_unicode_columns.py

Adds SentenceA_unicode and SentenceB_unicode columns to a pairs file
(Excel or CSV) by looking up each EWTS sentence in a Tibetan unicode corpus.

Two corpus formats are supported (auto-detected by extension):

  CSV  (e.g. merged_kangyur_tengyur_segments_v4.csv)
       Matching: SentenceA/SentenceB  →  Segmented_Text_EWTS  →  Segmented_Text
       Read in chunks; stops early once all sentences are resolved.

  Excel (e.g. all_gold_pairs_1000_scored_unicode.xlsx)
       The file already has SentenceA/SentenceB (EWTS) paired with
       SentenceA_unicode/SentenceB_unicode.  Both column pairs contribute
       to the same {ewts: unicode} lookup map.

Fallback: sentences not found in the corpus keep the original EWTS text.

Usage:
    python -m sub_tasks.add_unicode_columns --input_file path/to/pairs.xlsx
    python -m sub_tasks.add_unicode_columns \
        --input_file path/to/pairs.xlsx \
        --corpus_file path/to/all_gold_pairs_1000_scored_unicode.xlsx \
        --output_file path/to/pairs_with_unicode.xlsx
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CORPUS = "./data/merged_kangyur_tengyur_segments_v4.csv"
EWTS_COL = "Segmented_Text_EWTS"
UNICODE_COL = "Segmented_Text"


def build_ewts_unicode_map(corpus_path: str, needed: set[str]) -> dict[str, str]:
    """
    Scan the corpus CSV in chunks and return an {ewts: unicode} dict
    for all sentences in *needed*.  Stops early once every sentence is found.
    """
    mapping: dict[str, str] = {}
    remaining = set(needed)

    print(f"Building EWTS→unicode map for {len(needed)} unique sentences from {corpus_path}")

    chunks = pd.read_csv(
        corpus_path,
        usecols=[EWTS_COL, UNICODE_COL],
        chunksize=100_000,
        dtype=str,
    )

    for chunk in chunks:
        chunk[EWTS_COL] = chunk[EWTS_COL].str.strip()
        hits = chunk[chunk[EWTS_COL].isin(remaining)]
        for ewts, uni in zip(hits[EWTS_COL], hits[UNICODE_COL]):
            mapping[ewts] = uni
            remaining.discard(ewts)
        if not remaining:
            break

    if remaining:
        print(
            f"Warning: {len(remaining)} sentence(s) had no unicode match; "
            "EWTS text will be used as fallback."
        )

    return mapping


def build_ewts_unicode_map_from_pairs(corpus_path: str, needed: set[str]) -> dict[str, str]:
    """
    Build an {ewts: unicode} map from an Excel pairs file that already contains
    SentenceA / SentenceB (EWTS) alongside SentenceA_unicode / SentenceB_unicode.
    Both column pairs are used so every seen sentence contributes to the map.
    """
    df = pd.read_excel(corpus_path, dtype=str)

    for col in ("SentenceA", "SentenceB", "SentenceA_unicode", "SentenceB_unicode"):
        if col not in df.columns:
            raise ValueError(f"Corpus file is missing required column: '{col}'")

    mapping: dict[str, str] = {}
    for ewts_col, uni_col in (("SentenceA", "SentenceA_unicode"), ("SentenceB", "SentenceB_unicode")):
        sub = df[[ewts_col, uni_col]].dropna()
        sub = sub[sub[ewts_col].isin(needed)]
        mapping.update(zip(sub[ewts_col], sub[uni_col]))

    unresolved = needed - mapping.keys()
    print(f"Built EWTS→unicode map from pairs file: {len(mapping)} matches, {len(unresolved)} unresolved")
    if unresolved:
        print(f"Warning: {len(unresolved)} sentence(s) had no unicode match; EWTS text will be used as fallback.")

    return mapping


def _is_excel(path: str) -> bool:
    return Path(path).suffix.lower() in {".xlsx", ".xls"}


def add_unicode_columns(input_file: str, corpus_file: str, output_file: str) -> None:
    input_path = Path(input_file)

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path, dtype=str)

    for col in ("SentenceA", "SentenceB"):
        if col not in df.columns:
            raise ValueError(f"Input file is missing required column: '{col}'")

    needed = set(df["SentenceA"].dropna()) | set(df["SentenceB"].dropna())
    if _is_excel(corpus_file):
        ewts_map = build_ewts_unicode_map_from_pairs(corpus_file, needed)
    else:
        ewts_map = build_ewts_unicode_map(corpus_file, needed)

    df["SentenceA_unicode"] = df["SentenceA"].map(lambda x: ewts_map.get(x, x))
    df["SentenceB_unicode"] = df["SentenceB"].map(lambda x: ewts_map.get(x, x))

    out_path = Path(output_file)
    if out_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows with unicode columns → {out_path}")


def add_unicode_columns_folder(folder: str, corpus_file: str) -> None:
    """Process all .xlsx/.xls/.csv files in a folder in-place."""
    folder_path = Path(folder)
    files = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
    ]
    if not files:
        print(f"No Excel/CSV files found in {folder_path}")
        return

    print(f"Processing {len(files)} file(s) in {folder_path}")
    for f in files:
        print(f"\n--- {f.name} ---")
        try:
            add_unicode_columns(str(f), corpus_file, str(f))
        except Exception as e:
            print(f"  Skipped ({e})")


def main():
    parser = argparse.ArgumentParser(
        description="Add SentenceA_unicode / SentenceB_unicode columns to a pairs file or folder."
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Path to the input pairs file (.xlsx or .csv), or a folder to process all files in it.",
    )
    parser.add_argument(
        "--corpus_file", default=DEFAULT_CORPUS,
        help=f"Path to the Tibetan corpus (CSV or Excel, default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--output_file", default=None,
        help="Output file path (ignored when input_file is a folder). Defaults to overwriting the input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if input_path.is_dir():
        if args.output_file:
            print("Warning: --output_file is ignored when --input_file is a folder.")
        add_unicode_columns_folder(args.input_file, args.corpus_file)
    else:
        output_file = args.output_file if args.output_file else args.input_file
        add_unicode_columns(args.input_file, args.corpus_file, output_file)


if __name__ == "__main__":
    main()