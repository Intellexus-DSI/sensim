import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sub_tasks.aggregate_sets import aggregate_sets

_PUBLISH_DIR = Path('/Users/shaycohen/Repositories/Intellexus-model/sensim/results/publish-2026-03-01')

_MODEL_DISPLAY_NAMES = {
    'mbert-tibetan-continual-wylie-final': 'Tib-B',
    'tibetan-mbert-v1-anchor-positive': 'Tib-B-AP',
    'tibetan-mbert-v4-anchor-anchor': 'Tib-B-AA',
    'tibetan-mbert-v4-b64-consecutive-segments': 'Tib-B-CS',
    'tibetan-modernbert-anchor-positive': 'Tib-MB-AP',
    'tibetan-modernbert-v2-b64-anchor-anchor': 'Tib-MB-AP',
    'tibetan-modernbert-v4-anchor-anchor': 'Tib-MB-AA',
    'tibetan-modernbert-v4-b64-consecutive-segments': 'Tib-MB-CS',
    'IntellexusBert-2.0': 'Tib-MB',
    'cino-large-v2': 'CINO',
    'tibetan-roberta-base': 'T-Roberta',
}

#_EXCLUDED_MODELS = {'Tibetan-BERT-wwm', 'tibetan-mbert-v1-anchor-positive', 'tibetan-modernbert-v2-b64-anchor-anchor'}
_EXCLUDED_MODELS = {'Tibetan-BERT-wwm', "tibetan-modernbert-v2-b64-anchor-anchor", 'tibetan-mbert-v4-anchor-anchor', 'tibetan-modernbert-v4-anchor-anchor'}

_TRAIN_FILENAME_LABELS = {
    'train_pairs_X_shuffled_600_scored.xlsx': 'gold',
    'train_pairs_X_shuffled_600_scored.x': 'gold',  # truncated variant stored in some CSVs
    'merged_trainset_2026-02-26_16-02-12.xlsx': 'similarity',
    'merged_trainset_2026-02-27_12-24-40.xlsx': 'similarity',
    'merged_trainset_2026-02-26_17-38-47.xlsx': 'bws',
    'merged_trainset_2026-02-27_22-20-37.xlsx': 'bws',
    'merged_trainset_2026-02-26_17-38-47_2026-02-26_16-02-12.xlsx': 'emr+bws',
    'merged_trainset_2026-02-27_22-20-37-2026-02-27_12-24-40.xlsx': 'emr+bws',
}

_SOURCES = [
    (_PUBLISH_DIR / 'active_sampling_results_2026-02-27_12-24-40.xlsx', 'Tib-B-AP-EMR'),
    (_PUBLISH_DIR / 'active_sampling_results_2026-02-27_22-20-37.xlsx', 'Tib-B-AP-BWS'),
    (_PUBLISH_DIR / 'active_sampling_results_2026-03-03_17-26-53.xlsx', 'Tib-B-EMR'),
]

_METRICS = ['test_spearman', 'test_pearson']
_METRIC_LABELS = {'test_spearman': 'Spearman', 'test_pearson': 'Pearson'}
_LINE_STYLES = ['-', '--', ':']
_COLORS = ['tab:blue', 'tab:orange', 'tab:green']


def rename_train_filenames(filepath):
    df = pd.read_csv(filepath)
    print("Columns:", list(df.columns))
    train_filename_cols = [c for c in df.columns if c.startswith('train') and c.endswith('_filename')]
    for col in train_filename_cols:
        def _label(v):
            parts = [p.strip() for p in str(v).split('+')]
            return ' + '.join(_TRAIN_FILENAME_LABELS.get(p, p) for p in parts)
        df[col] = df[col].map(_label)
    df.to_csv(filepath, index=False)
    print(f"Renamed train filenames in: {filepath}")


def concat_no_fit_files(files, out_path):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        keep_cols = ['model_name_or_path', 'use_unicode_columns'] + [c for c in df.columns if 'spearman' in c or 'pearson' in c]
        dfs.append(df[[c for c in keep_cols if c in df.columns]])
    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
    print(f"Result: {len(result)} rows\n")
    display(result)


def generate_latex_table(publish_dir=None):
    if publish_dir is None:
        publish_dir = _PUBLISH_DIR
    publish_dir = Path(publish_dir)

    nf_df  = pd.concat([
        pd.read_csv(publish_dir / 'llms_aggregated_no_fit.csv'),
        pd.read_csv(publish_dir / 'llms_aggregated_no_fit_unicode.csv'),
    ], ignore_index=True)
    sim_df = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_similarity.csv')
    bws_df = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_bws.csv')

    metric_cols = ['test_spearman_mean', 'test_spearman_std', 'test_pearson_mean', 'test_pearson_std']

    def _select(df, train1, train):
        if 'model_short' not in df.columns:
            df = df.copy()
            df['model_short'] = df['model_name_or_path'].str.split('/').str[-1]
        mask = pd.Series(True, index=df.index)
        if 'train1_filename' in df.columns:
            mask &= df['train1_filename'].isna() if train1 is None else df['train1_filename'] == train1
        if 'train_filename' in df.columns and train is not None:
            mask &= df['train_filename'] == train
        sub = df[mask]
        agg = sub.groupby('model_short')[metric_cols].mean()
        if 'n_rows' in sub.columns:
            agg['n_rows'] = sub.groupby('model_short')['n_rows'].min()
        return agg

    type_configs = [
        ('baseline',               nf_df,  None,         None),
        ('gold',                   sim_df, None,         'gold'),
        ('emr',                    sim_df, None,         'similarity'),
        ('bws',                    bws_df, None,         'bws'),
        ('gold+sim',               sim_df, None,         'gold + similarity'),
        ('sim$\\to$gold',          sim_df, 'gold',       'similarity'),
        ('gold$\\to$sim',          sim_df, 'similarity', 'gold'),
        ('bws$\\to$gold',          bws_df, 'gold',       'bws'),
        ('gold$\\to$bws',          bws_df, 'bws',        'gold'),
    ]

    frames = {label: _select(df, t1, t) for label, df, t1, t in type_configs}
    combined = pd.concat(frames, axis=1)
    combined.columns.names = ['type', 'metric']

    def fmt(mean, std):
        if pd.isna(mean):
            return '--'
        if pd.isna(std):
            std = 0.0
        return f'${mean:.3f}\\pm{f"{std:.3f}"[1:]}$'

    types = [label for label, *_ in type_configs]
    n_types = len(types)
    col_spec = 'l' + 'rr' * n_types

    h1 = 'Model' + ''.join(f' & \\multicolumn{{2}}{{c}}{{{t}}}' for t in types) + ' \\\\'
    cmidrules = ' '.join(f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(n_types))
    h2 = ' & Sp. & Pe.' * n_types + ' \\\\'

    unicode_models = set(nf_df[nf_df['use_unicode_columns'].astype(str).str.lower() == 'true']['model_short'])
    sorted_models = sorted((m for m in combined.index if m not in _EXCLUDED_MODELS), key=lambda m: (0 if m in unicode_models else 1, _MODEL_DISPLAY_NAMES.get(m, m)))
    rows = []
    for i, model in enumerate(sorted_models):
        if i > 0 and model not in unicode_models and sorted_models[i - 1] in unicode_models:
            rows.append('\\midrule')
        display_name = _MODEL_DISPLAY_NAMES.get(model, model).replace('_', '\\_')
        parts = [display_name]
        for t in types:
            n_rows_val = combined.loc[model, (t, 'n_rows')] if (t, 'n_rows') in combined.columns else float('nan')
            if pd.notna(n_rows_val) and n_rows_val < 4:
                cell = f'X (n={int(n_rows_val)})'
                parts += [cell, cell]
            else:
                sp_mean = combined.loc[model, (t, 'test_spearman_mean')]
                sp_std  = combined.loc[model, (t, 'test_spearman_std')]
                pe_mean = combined.loc[model, (t, 'test_pearson_mean')]
                pe_std  = combined.loc[model, (t, 'test_pearson_std')]
                parts += [fmt(sp_mean, sp_std), fmt(pe_mean, pe_std)]
        rows.append(' & '.join(parts) + ' \\\\')

    latex = '\n'.join([
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
        h1,
        cmidrules,
        h2,
        '\\midrule',
        *rows,
        '\\bottomrule',
        '\\end{tabular}',
        '\\caption{Results}',
        '\\label{tab:results}',
        '\\end{table*}',
    ])

    out_path = publish_dir / 'results_table.tex'
    out_path.write_text(latex)
    print(f"LaTeX table saved to: {out_path}")
    print(latex)


def generate_latex_table_v2(publish_dir=None):
    if publish_dir is None:
        publish_dir = _PUBLISH_DIR
    publish_dir = Path(publish_dir)

    nf_df  = pd.concat([
        pd.read_csv(publish_dir / 'llms_aggregated_no_fit.csv'),
        pd.read_csv(publish_dir / 'llms_aggregated_no_fit_unicode.csv'),
    ], ignore_index=True)
    sim_df = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_similarity.csv')
    bws_df = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_bws.csv')

    metric_cols = ['test_spearman_mean', 'test_spearman_std']

    def _select(df, train1, train):
        if 'model_short' not in df.columns:
            df = df.copy()
            df['model_short'] = df['model_name_or_path'].str.split('/').str[-1]
        mask = pd.Series(True, index=df.index)
        if 'train1_filename' in df.columns:
            mask &= df['train1_filename'].isna() if train1 is None else df['train1_filename'] == train1
        if 'train_filename' in df.columns and train is not None:
            mask &= df['train_filename'] == train
        sub = df[mask]
        agg = sub.groupby('model_short')[metric_cols].mean()
        if 'n_rows' in sub.columns:
            agg['n_rows'] = sub.groupby('model_short')['n_rows'].min()
        return agg

    type_configs = [
        ('baseline',               nf_df,  None,         None),
        ('gold',                   sim_df, None,         'gold'),
        ('emr',                    sim_df, None,         'similarity'),
        ('bws',                    bws_df, None,         'bws'),
        ('gold+sim',               sim_df, None,         'gold + similarity'),
        ('sim$\\to$gold',          sim_df, 'gold',       'similarity'),
        ('gold$\\to$sim',          sim_df, 'similarity', 'gold'),
        ('bws$\\to$gold',          bws_df, 'gold',       'bws'),
        ('gold$\\to$bws',          bws_df, 'bws',        'gold'),
    ]

    frames = {label: _select(df, t1, t) for label, df, t1, t in type_configs}
    combined = pd.concat(frames, axis=1)
    combined.columns.names = ['type', 'metric']

    def fmt(mean, std):
        if pd.isna(mean):
            return '--'
        if pd.isna(std):
            std = 0.0
        return f'${mean:.3f}\\pm{f"{std:.2f}"[1:]}$'

    types = [label for label, *_ in type_configs]
    n_types = len(types)
    col_spec = 'l' + 'r' * n_types

    h1 = 'Model' + ''.join(f' & {t}' for t in types) + ' \\\\'

    unicode_models = set(nf_df[nf_df['use_unicode_columns'].astype(str).str.lower() == 'true']['model_short'])
    sorted_models = sorted((m for m in combined.index if m not in _EXCLUDED_MODELS), key=lambda m: (0 if m in unicode_models else 1, _MODEL_DISPLAY_NAMES.get(m, m)))
    rows = []
    for i, model in enumerate(sorted_models):
        if i > 0 and model not in unicode_models and sorted_models[i - 1] in unicode_models:
            rows.append('\\midrule')
        display_name = _MODEL_DISPLAY_NAMES.get(model, model).replace('_', '\\_')
        parts = [display_name]
        for t in types:
            n_rows_val = combined.loc[model, (t, 'n_rows')] if (t, 'n_rows') in combined.columns else float('nan')
            if pd.notna(n_rows_val) and n_rows_val < 4:
                parts.append(f'X (n={int(n_rows_val)})')
            else:
                sp_mean = combined.loc[model, (t, 'test_spearman_mean')]
                sp_std  = combined.loc[model, (t, 'test_spearman_std')]
                parts.append(fmt(sp_mean, sp_std))
        rows.append(' & '.join(parts) + ' \\\\')

    latex = '\n'.join([
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
        h1,
        '\\midrule',
        *rows,
        '\\bottomrule',
        '\\end{tabular}',
        '\\caption{Spearman correlation results}',
        '\\label{tab:results-spearman}',
        '\\end{table*}',
    ])

    out_path = publish_dir / 'results_table_v2.tex'
    out_path.write_text(latex)
    print(f"LaTeX table saved to: {out_path}")
    print(latex)


def generate_latex_table_v3(publish_dir=None):
    if publish_dir is None:
        publish_dir = _PUBLISH_DIR
    publish_dir = Path(publish_dir)

    nf_df  = pd.read_csv(publish_dir / 'llms_aggregated_no_fit_concat.csv')
    if 'model_short' not in nf_df.columns:
        nf_df['model_short'] = nf_df['model_name_or_path'].str.split('/').str[-1]
    sim_df    = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_similarity.csv')
    bws_df    = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_bws.csv')
    simbws_df = pd.read_csv(publish_dir / 'llms_aggregated_sets_results_simbws.csv')
    all_df    = pd.concat([sim_df, bws_df, simbws_df], ignore_index=True)

    metric_cols = ['test_spearman_mean', 'test_spearman_std']

    def _select(df, train1, train):
        if 'model_short' not in df.columns:
            df = df.copy()
            df['model_short'] = df['model_name_or_path'].str.split('/').str[-1]
        mask = pd.Series(True, index=df.index)
        if 'train1_filename' in df.columns:
            mask &= df['train1_filename'].isna() if train1 is None else df['train1_filename'] == train1
        if 'train_filename' in df.columns and train is not None:
            mask &= df['train_filename'] == train
        sub = df[mask]
        agg = sub.groupby('model_short')[metric_cols].mean()
        if 'n_rows' in sub.columns:
            agg['n_rows'] = sub.groupby('model_short')['n_rows'].min()
        return agg

    def fmt(mean, std, bold=False):
        if pd.isna(mean):
            return '--'
        mean_str = f'\\mathbf{{{mean:.3f}}}' if bold else f'{mean:.3f}'
        if pd.isna(std):
            std = 0.0
        return f'${mean_str}\\pm{f"{std:.2f}"[1:]}$'

    all_type_configs = [
        ('baseline',               nf_df,     None,         None),
        ('gold',                   all_df,    None,         'gold'),
        ('emr',                    sim_df,    None,         'similarity'),
        ('bws',                    bws_df,    None,         'bws'),
        ('gold+emr',               sim_df,    None,         'gold + similarity'),
        ('gold+bws',               bws_df,    None,         'gold + bws'),
        ('emr$\\to$gold',          sim_df,    'gold',       'similarity'),
        ('gold$\\to$emr',          sim_df,    'similarity', 'gold'),
        ('bws$\\to$gold',          bws_df,    'gold',       'bws'),
        ('gold$\\to$bws',          bws_df,    'bws',        'gold'),
        ('emr+bws',                simbws_df, None,         'emr+bws'),
        ('gold$\\to$emr+bws',      simbws_df, 'emr+bws',    'gold'),
        ('emr+bws$\\to$gold',      simbws_df, 'gold',       'emr+bws'),
    ]

    frames = {label: _select(df, t1, t) for label, df, t1, t in all_type_configs}
    combined = pd.concat(frames, axis=1)
    combined.columns.names = ['type', 'metric']
    unicode_models = set(nf_df[nf_df['use_unicode_columns'].astype(str).str.lower() == 'true']['model_short'])
    models = sorted((m for m in combined.index if m not in _EXCLUDED_MODELS), key=lambda m: (0 if m in unicode_models else 1, _MODEL_DISPLAY_NAMES.get(m, m)))

    def _make_table(type_subset, caption, label, note=None):
        n = len(type_subset)
        col_spec = 'l' + 'r' * n
        h1 = 'Model' + ''.join(f' & {t}' for t in type_subset) + ' \\\\'
        best_model = {
            t: combined.loc[:, (t, 'test_spearman_mean')].idxmax()
            for t in type_subset
        }
        rows = []
        for i, model in enumerate(models):
            if i > 0 and model not in unicode_models and models[i - 1] in unicode_models:
                rows.append('\\midrule')
            display_name = _MODEL_DISPLAY_NAMES.get(model, model).replace('_', '\\_')
            if display_name.startswith('Tib-MB'):
                display_name += '\\footref{fn:inhouse_modernbert_model}'
            elif display_name.startswith('Tib-B'):
                display_name += '\\footref{fn:inhouse_bert_model}'
            parts = [display_name]
            for t in type_subset:
                n_rows_val = combined.loc[model, (t, 'n_rows')] if (t, 'n_rows') in combined.columns else float('nan')
                if pd.notna(n_rows_val) and n_rows_val < 4:
                    parts.append(f'X (n={int(n_rows_val)})')
                else:
                    sp_mean = combined.loc[model, (t, 'test_spearman_mean')]
                    sp_std  = combined.loc[model, (t, 'test_spearman_std')]
                    parts.append(fmt(sp_mean, sp_std, bold=(model == best_model[t])))
            rows.append(' & '.join(parts) + ' \\\\')
        note_lines = [f'\\footnotesize{{\\textit{{{note}}}}}'] if note else []
        return '\n'.join([
            '\\begin{table*}[t]',
            '\\centering',
            '\\small',
            f'\\begin{{tabular}}{{{col_spec}}}',
            '\\toprule',
            h1,
            '\\midrule',
            *rows,
            '\\bottomrule',
            '\\end{tabular}',
            *note_lines,
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\end{table*}',
        ])

    types = [label for label, *_ in all_type_configs]
    table1 = _make_table(
        types[:5],
        'Spearman correlation results (baseline and single-dataset fine-tuning)',
        'tab:results-spearman-1',
    )
    table2 = _make_table(
        types[5:10],
        'Spearman correlation results (combined and sequential fine-tuning)',
        'tab:results-spearman-2',
    )
    table3 = _make_table(
        types[10:],
        'Spearman correlation results (emr+bws merged fine-tuning)',
        'tab:results-spearman-3',
    )
    table4 = _make_table(
        ['baseline', 'gold', 'emr', 'bws', 'emr$\\to$gold', 'bws$\\to$gold', 'emr+bws$\\to$gold'],
        '\\textbf{Spearman $\\rho$} correlation results (gold fine-tuning summary)',
        'tab:results-summary',
        note='\\textit{\\textbf{AP}: Unsupervised with anchor-positive and Multiple Negatives Ranking Loss.} \\\\ \\textbf{CS}:\\textit{Unsupervised with consecutive-segments and Multiple Negatives Ranking Loss.}',
    )

    latex = table1 + '\n\n' + table2 + '\n\n' + table3
    out_path = publish_dir / 'results_table_v3.tex'
    out_path.write_text(latex)
    print(f"LaTeX table saved to: {out_path}")
    print(latex)

    summary_path = publish_dir / 'results_summary.tex'
    summary_path.write_text(table4)
    print(f"LaTeX table saved to: {summary_path}")
    print(table4)


def generate_publish_table(use_5000: bool = False):
    publish_dir = _PUBLISH_DIR

    # No-fit baseline files — use pre-computed if available, else aggregate from raw
    no_fit_sets = [
        ('llms_no_fit.csv',                    'llms_aggregated_no_fit.csv'),
        ('llms_no_fit_unicode.csv',            'llms_aggregated_no_fit_unicode.csv'),
        ('llms_no_fit_2026_03_09.csv',         'llms_aggregated_no_fit_2026_03_09.csv'),
        ('llms_no_fit_unicode_2026_03_09.csv', 'llms_aggregated_no_fit_unicode_2026_03_09.csv'),
    ]
    for sets_file, agg_file in no_fit_sets:
        agg_path = publish_dir / agg_file
        if not agg_path.exists():
            aggregate_sets(publish_dir / sets_file, agg_path)

    # Raw timestamped results → named aggregated files
    if use_5000:
        sets_results_copies = [
            ('llms_aggregated_sets_results_2026-02-27_12-24-40_5000.csv',                      'llms_aggregated_sets_results_similarity.csv'),
            ('llms_aggregated_sets_results_2026-02-27_22-20-37_5000.csv',                      'llms_aggregated_sets_results_bws.csv'),
            ('llms_aggregated_sets_results_2026-02-27_22-20-37-2026-02-27_12-24-40_10000.csv', 'llms_aggregated_sets_results_simbws.csv'),
        ]
    else:
        sets_results_copies = [
            ('llms_aggregated_sets_results_2026-02-26_16-02-12_2500.csv',                     'llms_aggregated_sets_results_similarity.csv'),
            ('llms_aggregated_sets_results_2026-02-27_22-20-37_2500.csv',                     'llms_aggregated_sets_results_bws.csv'),
            ('llms_aggregated_sets_results_2026-02-26_17-38-47_2026-02-26_16-02-12_5000.csv', 'llms_aggregated_sets_results_simbws.csv'),
        ]
    for src, dst in sets_results_copies:
        shutil.copy(publish_dir / src, publish_dir / dst)
        rename_train_filenames(publish_dir / dst)

    concat_no_fit_files(
        files=[publish_dir / agg for _, agg in no_fit_sets],
        out_path=publish_dir / 'llms_aggregated_no_fit_concat.csv',
    )

    generate_latex_table(publish_dir)
    generate_latex_table_v2(publish_dir)
    generate_latex_table_v3(publish_dir)


def _plot_line_with_n_rows(ax, x, y, n_rows, color, label, linestyle='-', marker=None, threshold=4):
    """Plot a line, replacing points where n_rows < threshold with 'X\n<n>' annotations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if n_rows is not None:
        n_rows = np.asarray(n_rows)
        low_mask = n_rows < threshold
    else:
        low_mask = np.zeros(len(x), dtype=bool)

    y_plot = y.copy()
    y_plot[low_mask] = np.nan

    ax.plot(x, y_plot, color=color, linestyle=linestyle, marker=marker, label=label)

    for xi, yi, ni in zip(x[low_mask], y[low_mask], (n_rows[low_mask] if n_rows is not None else [])):
        if not np.isnan(yi):
            ax.text(xi, yi, f'X\n{int(ni)}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')


def plot_embedding_drift(out_path=None):
    if out_path is None:
        out_path = _PUBLISH_DIR / 'embedding_drift.pdf'

    fig, ax = plt.subplots(figsize=(7, 4))

    for (filepath, label), color in zip(_SOURCES, _COLORS):
        df = pd.read_excel(filepath)
        cols = ['iteration', 'embedding_drift'] + (['n_rows'] if 'n_rows' in df.columns else [])
        data = df[cols].dropna(subset=['iteration', 'embedding_drift'])
        n_rows = data['n_rows'].values if 'n_rows' in data.columns else None
        _plot_line_with_n_rows(ax, data['iteration'], data['embedding_drift'], n_rows, color, label, marker='o')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Embedding Drift')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved to: {out_path}')


def plot_test_metrics(out_path=None):
    if out_path is None:
        out_path = _PUBLISH_DIR / 'test_metrics.pdf'

    fig, ax = plt.subplots(figsize=(7, 4))

    for (filepath, label), color in zip(_SOURCES, _COLORS):
        df = pd.read_excel(filepath)
        for metric, ls in zip(_METRICS, _LINE_STYLES):
            cols = ['iteration', metric] + (['n_rows'] if 'n_rows' in df.columns else [])
            data = df[cols].dropna(subset=['iteration', metric])
            n_rows = data['n_rows'].values if 'n_rows' in data.columns else None
            _plot_line_with_n_rows(
                ax, data['iteration'], data[metric], n_rows, color,
                label=f'{label} – {_METRIC_LABELS[metric]}', linestyle=ls,
            )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spearman/Pearson Correlation')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved to: {out_path}')


def plot_faiss_distance(out_path=None):
    if out_path is None:
        out_path = _PUBLISH_DIR / 'faiss_distance_mean.pdf'

    fig, ax = plt.subplots(figsize=(7, 4))

    for (filepath, label), color in zip(_SOURCES, _COLORS):
        df = pd.read_excel(filepath)
        cols = ['iteration', 'faiss_distance_mean', 'faiss_distance_std'] + (['n_rows'] if 'n_rows' in df.columns else [])
        data = df[cols].dropna(subset=['iteration', 'faiss_distance_mean', 'faiss_distance_std'])
        n_rows = data['n_rows'].values if 'n_rows' in data.columns else None
        _plot_line_with_n_rows(ax, data['iteration'], data['faiss_distance_mean'], n_rows, color, label, marker='o')
        valid = data if n_rows is None else data[np.asarray(n_rows) >= 4]
        ax.fill_between(
            valid['iteration'],
            valid['faiss_distance_mean'] - valid['faiss_distance_std'],
            valid['faiss_distance_mean'] + valid['faiss_distance_std'],
            color=color, alpha=0.2,
        )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('FAISS Distance Mean')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved to: {out_path}')


def publish_plots():
    plot_embedding_drift()
    plot_test_metrics()
    plot_faiss_distance()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-5000', action='store_true', dest='use_5000',
                        help='Use 5000-sample source files instead of 2500')
    args = parser.parse_args()
    generate_publish_table(use_5000=args.use_5000)
    publish_plots()
