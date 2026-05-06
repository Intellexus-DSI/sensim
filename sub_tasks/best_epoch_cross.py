"""
best_epoch_cross.py

Reads an already-aggregated cross epochs CSV and keeps only the best epoch
(highest validation_spearman_mean) per (model_name_or_path, train1_filename,
train_filename, no_fit) combination.  Column order is preserved.

Optional --summarize flag writes a second CSV with one row per model and one
column-pair (test_spearman_mean / test_spearman_std) per unique setting.

Optional --tex_file flag writes a LaTeX threeparttable.  Requires
--gold_file and --synthetic_file substrings to classify settings as:
  Baseline (no_fit=True) | Gold | Synthetic | Gold→S | Synthetic→G

Usage:
    python sub_tasks/best_epoch_cross.py \\
        --results_file results/.../llms_cross_aggregated_sets_results_*_epochs.csv \\
        [--summarize] \\
        [--gold_file train_pairs --synthetic_file merged_trainset --tex_file table.tex]
"""

import argparse
from pathlib import Path

import pandas as pd

GROUP_COLS = ['model_name_or_path', 'train1_filename', 'train_filename', 'no_fit']
SUMMARY_SETTING_COLS = ['train1_filename', 'train_filename', 'no_fit']

# Canonical setting keys in display order
_SETTING_ORDER = ['baseline', 'gold', 'synthetic', 'gold_to_s', 's_to_gold']
_SETTING_HEADER = {
    'baseline':  'Baseline',
    'gold':      'Gold',
    'synthetic': 'Synthetic',
    'gold_to_s': r'Gold$\to$S\tnote{a}',
    's_to_gold': r'Synthetic$\to$G\tnote{b}',
}


def find_best_epochs(results_file: str) -> pd.DataFrame:
    df = pd.read_csv(results_file)

    df['validation_spearman_mean'] = pd.to_numeric(df['validation_spearman_mean'], errors='coerce')

    idx_best = df.groupby(GROUP_COLS, dropna=False)['validation_spearman_mean'].idxmax()
    best_df = df.loc[idx_best].reset_index(drop=True)

    best_df = best_df[df.columns]  # restore original column order

    return best_df.sort_values('validation_spearman_mean', ascending=False).reset_index(drop=True)


def _classify_setting(row, gold_pat: str, synth_pat: str) -> str | None:
    if str(row.get('no_fit', '')).strip().lower() == 'true':
        return 'baseline'
    t1_raw = row.get('train1_filename')
    t_raw  = row.get('train_filename')
    t1 = '' if pd.isna(t1_raw) else str(t1_raw).strip()
    t  = '' if pd.isna(t_raw)  else str(t_raw).strip()
    if not t1:  # single-phase
        if gold_pat  in t: return 'gold'
        if synth_pat in t: return 'synthetic'
    else:       # two-phase: train_filename is first, train1_filename is the continuation (→)
        if gold_pat  in t and synth_pat in t1: return 'gold_to_s'
        if synth_pat in t and gold_pat  in t1: return 's_to_gold'
    return None


def build_tex(best_df: pd.DataFrame, gold_pat: str, synth_pat: str,
              caption: str, label: str,
              model_names: dict[str, str] | None = None) -> str:
    df = best_df.copy()
    df['_setting'] = df.apply(_classify_setting, axis=1,
                              gold_pat=gold_pat, synth_pat=synth_pat)

    settings_present = [s for s in _SETTING_ORDER if s in df['_setting'].values]

    # model → setting → (mean, std)
    table: dict[str, dict[str, tuple]] = {}
    model_short_map: dict[str, str] = {}
    for _, row in df.iterrows():
        m = row['model_name_or_path']
        s = row['_setting']
        if s is None:
            continue
        table.setdefault(m, {})[s] = (
            row.get('test_spearman_mean'),
            row.get('test_spearman_std'),
        )
        short = row.get('model_short', m.split('/')[-1])
        model_short_map[m] = (model_names or {}).get(short, short)

    # column-wise max for bolding
    col_max: dict[str, float] = {}
    for s in settings_present:
        vals = [table[m][s][0] for m in table if s in table[m]
                and table[m][s][0] is not None]
        if vals:
            col_max[s] = max(vals)

    n_cols = 1 + len(settings_present)
    col_spec = 'l' * n_cols
    header = ' & '.join(['Model'] + [_SETTING_HEADER[s] for s in settings_present])

    rows = []
    for m, short in model_short_map.items():
        cells = [short]
        for s in settings_present:
            if s in table.get(m, {}):
                mean, std = table[m][s]
                cell = f'{mean:.3f} $\\pm$ {std:.2f}'
                if col_max.get(s) is not None and abs(mean - col_max[s]) < 1e-9:
                    cell = r'\bfseries ' + cell
            else:
                cell = '—'
            cells.append(cell)
        rows.append(' & '.join(cells) + r' \\')

    has_ab_notes = any(s in settings_present for s in ('gold_to_s', 's_to_gold'))

    lines = [
        r'\begin{table*}',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{threeparttable}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header + r' \\',
        r'\midrule',
        *rows,
        r'\bottomrule',
        r'\end{tabular}',
    ]
    if has_ab_notes:
        lines += [
            r'\begin{tablenotes}',
            r'    \small',
            r"    \item [a] Gold$\to$S: Gold$\to$Synthetic.",
            r"    \item [b] Synthetic$\to$G: Synthetic$\to$Gold.",
            r'\end{tablenotes}',
        ]
    lines += [
        r'\end{threeparttable}',
        r'\end{table*}',
    ]
    return '\n'.join(lines) + '\n'


def build_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot best-epoch results into one row per model, one column-pair per setting."""
    keep = GROUP_COLS + ['test_spearman_mean', 'test_spearman_std']
    df = best_df[[c for c in keep if c in best_df.columns]].copy()

    # Build a readable label for each setting
    def setting_label(row):
        parts = []
        if pd.notna(row.get('train1_filename')):
            parts.append(str(row['train1_filename']))
        if pd.notna(row.get('train_filename')):
            parts.append(str(row['train_filename']))
        if str(row.get('no_fit', '')).strip().lower() == 'true':
            parts.append('no_fit')
        return ' | '.join(parts) if parts else 'default'

    df['setting'] = df.apply(setting_label, axis=1)

    pivot_mean = df.pivot(index='model_name_or_path', columns='setting', values='test_spearman_mean')
    pivot_std  = df.pivot(index='model_name_or_path', columns='setting', values='test_spearman_std')

    pivot_mean.columns = [f"{c}_test_spearman_mean" for c in pivot_mean.columns]
    pivot_std.columns  = [f"{c}_test_spearman_std"  for c in pivot_std.columns]

    # Interleave mean/std columns per setting
    settings = df['setting'].unique()
    ordered_cols = []
    for s in settings:
        mean_col = f"{s}_test_spearman_mean"
        std_col  = f"{s}_test_spearman_std"
        if mean_col in pivot_mean.columns:
            ordered_cols.append(mean_col)
        if std_col in pivot_std.columns:
            ordered_cols.append(std_col)

    summary = pd.concat([pivot_mean, pivot_std], axis=1)[ordered_cols].reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Keep the best epoch per model/train-file combination from an aggregated epochs CSV.'
    )
    parser.add_argument(
        '--results_file',
        required=True,
        help='Path to the aggregated epochs CSV (e.g. llms_cross_aggregated_sets_results_*_epochs.csv)',
    )
    parser.add_argument(
        '--output_file',
        default=None,
        help='Output CSV path. Defaults to <input>_best_epoch.csv',
    )
    parser.add_argument(
        '--summarize',
        action='store_true',
        help='Also write a summary CSV with one row per model and one column-pair '
             '(test_spearman_mean / test_spearman_std) per unique setting. '
             'Saved next to --output_file as *_summary.csv',
    )
    parser.add_argument(
        '--tex_file',
        default=None,
        help='Write a LaTeX threeparttable to this path. '
             'Defaults to <output_file stem>_table.tex. '
             'Requires --gold_file and --synthetic_file.',
    )
    parser.add_argument(
        '--gold_file',
        default=None,
        help='Substring identifying the gold train filename (e.g. "train_pairs").',
    )
    parser.add_argument(
        '--synthetic_file',
        default=None,
        help='Substring identifying the synthetic train filename (e.g. "merged_trainset").',
    )
    parser.add_argument(
        '--caption',
        default='Cross-Encoder models',
        help='LaTeX table caption (default: "Cross-Encoder models").',
    )
    parser.add_argument(
        '--label',
        default='tab:cross_models',
        help='LaTeX \\label for the table (default: tab:cross_models).',
    )
    parser.add_argument(
        '--model_names',
        default=None,
        help='JSON mapping from model_short to display name, e.g. '
             '\'{"mbert-tibetan-continual-wylie-final": "mbert-Tibetan", '
             '"tibetan-roberta-base": "T-RoBERTa"}\'',
    )
    args = parser.parse_args()

    result = find_best_epochs(args.results_file)

    if result.empty:
        print("No rows found.")
        return

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    display_cols = [c for c in [
        'model_short', 'train1_filename', 'train_filename', 'no_fit', 'epoch',
        'validation_spearman_mean', 'validation_spearman_std',
        'test_spearman_mean', 'test_spearman_std',
    ] if c in result.columns]
    print(result[display_cols].to_string(index=False))

    output_file = args.output_file
    if output_file is None:
        p = Path(args.results_file)
        stem = p.stem
        if stem.endswith('_epochs'):
            stem = stem[: -len('_epochs')]
        output_file = str(p.with_name(stem + '_best_epoch.csv'))

    result.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    if args.summarize:
        summary = build_summary(result)
        summary_file = str(Path(output_file).with_name(Path(output_file).stem + '_summary.csv'))
        summary.to_csv(summary_file, index=False)
        print(f"\nSummary ({len(summary)} model(s)):")
        pd.set_option('display.max_colwidth', 60)
        print(summary.to_string(index=False))
        print(f"\nSummary saved to: {summary_file}")

    if args.tex_file is not None or args.gold_file or args.synthetic_file:
        if not args.gold_file or not args.synthetic_file:
            print("\nWARNING: --gold_file and --synthetic_file are both required to generate a TeX table. Skipping.")
        else:
            import json
            model_names = json.loads(args.model_names) if args.model_names else None
            tex = build_tex(result, args.gold_file, args.synthetic_file,
                            args.caption, args.label, model_names=model_names)
            tex_file = args.tex_file or str(
                Path(output_file).with_name(Path(output_file).stem + '_table.tex')
            )
            Path(tex_file).write_text(tex, encoding='utf-8')
            print(f"\nTeX table saved to: {tex_file}")


if __name__ == '__main__':
    main()
