#!/usr/bin/env python3
"""Q2 Spearman baseline; the main analysis uses all 227 measured features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def rank_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ar = StandardScaler().fit_transform(rankdata(a, axis=0))
    br = ar if a is b else StandardScaler().fit_transform(rankdata(b, axis=0))
    return ar.T @ br / len(a)


def diverse_top3(scores: np.ndarray, within: np.ndarray, penalty: float = 0.35) -> list[int]:
    chosen = []
    for _ in range(3):
        adjusted = np.abs(scores).copy()
        if chosen:
            adjusted -= penalty * np.max(np.abs(within[:, chosen]), axis=1)
        adjusted[chosen] = -np.inf
        chosen.append(int(np.argmax(adjusted)))
    return chosen


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def pairwise_detected_spearman(x: np.ndarray, detected: np.ndarray, minimum_n: int):
    p = x.shape[1]
    corr = np.eye(p)
    pval = np.zeros((p, p))
    count = np.full((p, p), len(x), dtype=int)
    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            keep = detected[:, i] & detected[:, j]
            n = int(keep.sum())
            count[i, j] = count[j, i] = n
            if n >= minimum_n:
                result = spearmanr(x[keep, i], x[keep, j])
                rho, pv = float(result.statistic), float(result.pvalue)
            else:
                rho, pv = np.nan, np.nan
            corr[i, j] = corr[j, i] = rho
            pval[i, j] = pval[j, i] = pv
            pairs.append((i, j))
    return corr, pval, count, pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--bootstrap', type=int, default=60)
    p.add_argument('--threshold', type=float, default=0.7)
    p.add_argument('--stage', choices=['network', 'top3', 'all'], default='all')
    p.add_argument('--feature-scope', choices=['original211', 'measured227'], default='measured227')
    p.add_argument('--correlation-source', choices=['published', 'common-detected'], default='published')
    p.add_argument('--min-common-detected', type=int, default=30)
    p.add_argument('--fdr-alpha', type=float, default=None)
    p.add_argument('--penalty', type=float, default=0.35)
    p.add_argument('--ridge-alpha', type=float, default=100.0)
    p.add_argument('--comparison', choices=['none', 'ridge'], default='ridge')
    p.add_argument('--seed', type=int, default=20261027)
    args = p.parse_args()
    root = args.root
    out = args.output or root / 'reports/modeling/q2_baseline_v1'
    out.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(root / 'data/interim/chem_feature_sets_v1.csv')
    if args.feature_scope == 'measured227':
        features = reg.loc[~reg.is_derived, 'feature'].tolist()
    else:
        features = reg.loc[reg.q2_primary_network, 'feature'].tolist()
    x_all = pd.read_csv(root / 'data/interim/chem_log10_matrix_v1.csv').set_index('beer')
    y_all = pd.read_csv(root / 'data/interim/sensory_targets_v1.csv').set_index('beer')
    status = pd.read_csv(root / 'data/interim/chem_observation_status_v1.csv').set_index('beer')
    published = pd.read_csv(root / 'data/raw/2.csv', index_col=0)
    # The 19 Greek-letter names in 8.xlsx are mojibake; positional mapping is
    # verified for the other 212 names before it is used for all 231 columns.
    names = pd.read_excel(root / 'data/raw/8.xlsx', sheet_name='Sensory thresholds').set_index('field_name_R')
    ordered = x_all.columns.tolist()
    matches = sum(str(names.loc[f, 'chemical_name']) == n for f, n in zip(ordered, published.columns))
    assert len(ordered) == published.shape[0] == published.shape[1] == 231 and matches == 212
    assert published.index.tolist() == published.columns.tolist()
    assert x_all.index.equals(y_all.index) and x_all.index.equals(status.index)
    expected_features = 227 if args.feature_scope == 'measured227' else 211
    assert len(features) == expected_features and not x_all[features].isna().any().any()
    corr = published.to_numpy(dtype=float)
    assert np.allclose(corr, corr.T) and np.allclose(np.diag(corr), 1)
    ixs = [ordered.index(f) for f in features]
    x = x_all[features].to_numpy(float)
    y = y_all.to_numpy(float)
    labels = y_all.columns.tolist()
    detected = status[features].eq(0).to_numpy()
    if args.correlation_source == 'common-detected':
        c, pair_p, pair_n, pairs = pairwise_detected_spearman(x, detected, args.min_common_detected)
    else:
        c = corr[np.ix_(ixs, ixs)]
        pair_p = np.full_like(c, np.nan)
        pair_n = np.full_like(c, len(x), dtype=int)
        pairs = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    pair_q = np.full_like(c, np.nan)
    valid_pairs = [(i, j) for i, j in pairs if np.isfinite(pair_p[i, j])]
    if valid_pairs:
        adjusted = bh_adjust(np.array([pair_p[i, j] for i, j in valid_pairs]))
        for (i, j), q in zip(valid_pairs, adjusted):
            pair_q[i, j] = pair_q[j, i] = q
    graph = nx.Graph()
    graph.add_nodes_from(features)
    edge_rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            rho = c[i, j]
            fdr_ok = args.fdr_alpha is None or (np.isfinite(pair_q[i, j]) and pair_q[i, j] < args.fdr_alpha)
            if np.isfinite(rho) and abs(rho) >= args.threshold and fdr_ok:
                graph.add_edge(features[i], features[j], weight=abs(rho), sign=np.sign(rho))
                edge_rows.append((i, j, rho, pair_p[i, j], pair_q[i, j], pair_n[i, j]))
    communities = list(nx.community.greedy_modularity_communities(graph, weight='weight'))
    module = {f: k for k, group in enumerate(communities) for f in group}
    pd.DataFrame({'feature': features, 'module': [module[f] for f in features],
                  'degree': [graph.degree(f) for f in features]}).to_csv(out / 'nodes.csv', index=False)
    rng = np.random.default_rng(args.seed)
    edge_count = np.zeros(len(edge_rows), dtype=int)
    edge_i = np.array([row[0] for row in edge_rows], dtype=int)
    edge_j = np.array([row[1] for row in edge_rows], dtype=int)
    for _ in range(args.bootstrap):
        rows = rng.integers(0, len(x), len(x))
        if args.correlation_source == 'published':
            boot_x = x[rows]
            boot_corr = rank_corr(boot_x, boot_x)
            edge_count += (np.abs(boot_corr[edge_i, edge_j]) >= args.threshold)
            continue
        for e, (i, j, _, _, _, _) in enumerate(edge_rows):
            if args.correlation_source == 'common-detected':
                keep = detected[rows, i] & detected[rows, j]
                if int(keep.sum()) < args.min_common_detected:
                    continue
                rho = spearmanr(x[rows[keep], i], x[rows[keep], j]).statistic
            else:
                rho = spearmanr(x[rows, i], x[rows, j]).statistic
            edge_count[e] += np.isfinite(rho) and abs(rho) >= args.threshold
    nd = status[features].eq(1).to_numpy(dtype=float)
    edge_output = []
    for e, (i, j, rho, pv, qv, common_n) in enumerate(edge_rows):
        phi = np.corrcoef(nd[:, i], nd[:, j])[0, 1] if nd[:, i].std() and nd[:, j].std() else np.nan
        edge_output.append({'source': features[i], 'target': features[j], 'rho': rho,
                            'p_value': pv, 'fdr_q_value': qv, 'common_detected_n': common_n,
                            'bootstrap_edge_frequency': edge_count[e] / args.bootstrap,
                            'shared_nondetect_phi': phi})
    pd.DataFrame(edge_output).to_csv(out / 'edges.csv', index=False)
    network_summary = {
        'stage': args.stage, 'feature_scope': args.feature_scope,
        'correlation_source': args.correlation_source,
        'nodes': len(features), 'edges': graph.number_of_edges(),
        'isolates': len(list(nx.isolates(graph))), 'components': nx.number_connected_components(graph),
        'communities': len(communities), 'largest_component': max(map(len, nx.connected_components(graph))),
        'largest_community': max(map(len, communities)), 'threshold': args.threshold,
        'fdr_alpha': args.fdr_alpha, 'min_common_detected': args.min_common_detected,
        'tested_pairs': len(valid_pairs) if args.correlation_source == 'common-detected' else len(pairs),
        'bootstrap': args.bootstrap, 'published_name_matches': matches,
        'seed': args.seed, 'redundancy_penalty': args.penalty,
        'comparison': args.comparison,
        'ridge_alpha': args.ridge_alpha if args.comparison == 'ridge' else None,
    }
    if args.stage == 'network':
        (out / 'summary.json').write_text(json.dumps(network_summary, indent=2), encoding='utf-8')
        print(out)
        return

    association = rank_corr(x, y)
    baseline = {label: diverse_top3(association[:, k], c, args.penalty)
                for k, label in enumerate(labels)}
    selections = np.zeros((len(features), len(labels)), dtype=int)
    signs = np.zeros_like(selections)
    for _ in range(args.bootstrap):
        rows = rng.integers(0, len(x), len(x))
        bx, by = x[rows], y[rows]
        bc = rank_corr(bx, bx)
        ba = rank_corr(bx, by)
        for k in range(len(labels)):
            for i in diverse_top3(ba[:, k], bc, args.penalty):
                selections[i, k] += 1
                signs[i, k] += np.sign(ba[i, k])
    ridge = None
    if args.comparison == 'ridge':
        sx = StandardScaler().fit_transform(x)
        sy = StandardScaler().fit_transform(y)
        ridge = Ridge(alpha=args.ridge_alpha).fit(sx, sy)
    rankings = []
    overlap = []
    for k, label in enumerate(labels):
        ridge_top = np.argsort(-np.abs(ridge.coef_[k]))[:3] if ridge is not None else np.array([], dtype=int)
        if ridge is not None:
            a, b = set(baseline[label]), set(ridge_top)
            overlap.append({'label': label, 'top3_jaccard_marginal_vs_ridge': len(a & b) / len(a | b)})
        for rank, i in enumerate(baseline[label], 1):
            rankings.append({'label': label, 'rank': rank, 'feature': features[i],
                             'published_graph_module': module[features[i]],
                             'spearman_to_sensory': association[i, k],
                             'bootstrap_top3_frequency': selections[i, k] / args.bootstrap,
                             'bootstrap_sign_agreement': abs(signs[i, k]) / max(1, selections[i, k]),
                             **({'ridge_top3': features[i] in [features[j] for j in ridge_top]}
                                if ridge is not None else {})})
    pd.DataFrame(rankings).to_csv(out / 'sensory_top3.csv', index=False)
    if overlap:
        pd.DataFrame(overlap).to_csv(out / 'method_overlap.csv', index=False)
    elif (out / 'method_overlap.csv').exists():
        (out / 'method_overlap.csv').unlink()
    (out / 'summary.json').write_text(json.dumps({
        **network_summary,
        'median_top3_frequency': float(np.median([r['bootstrap_top3_frequency'] for r in rankings])),
        'median_method_jaccard': (float(np.median([r['top3_jaccard_marginal_vs_ridge'] for r in overlap]))
                                  if overlap else None),
        'interpretation': 'Association only; graph edges and Top-3 are not causal effects.'
    }, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
