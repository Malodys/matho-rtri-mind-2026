#!/usr/bin/env python3
"""Generate submission figures from the canonical Q2 baseline outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


COLORS = {"positive": "#D55E00", "negative": "#0072B2"}


def save_network(result_dir: Path) -> None:
    nodes = pd.read_csv(result_dir / "nodes.csv")
    edges = pd.read_csv(result_dir / "edges.csv")
    rho_column = "rho_published" if "rho_published" in edges.columns else "rho"
    graph = nx.Graph()
    graph.add_nodes_from(nodes.feature)
    for row in edges.itertuples():
        rho = float(getattr(row, rho_column))
        graph.add_edge(row.source, row.target, rho=rho, weight=abs(rho))
    connected = graph.subgraph([node for node in graph if graph.degree(node) > 0]).copy()
    position = nx.spring_layout(connected, seed=20261027, weight="weight", k=0.55)
    degree = np.array([connected.degree(node) for node in connected], float)
    fig, ax = plt.subplots(figsize=(11, 8))
    positive = [(a, b) for a, b, data in connected.edges(data=True) if data["rho"] > 0]
    negative = [(a, b) for a, b, data in connected.edges(data=True) if data["rho"] < 0]
    nx.draw_networkx_edges(connected, position, edgelist=positive, edge_color=COLORS["positive"],
                           width=1.0, alpha=0.42, ax=ax)
    nx.draw_networkx_edges(connected, position, edgelist=negative, edge_color=COLORS["negative"],
                           width=1.2, alpha=0.65, ax=ax)
    nx.draw_networkx_nodes(connected, position, node_size=34 + 22 * degree,
                           node_color=degree, cmap="viridis", linewidths=0.4,
                           edgecolors="white", ax=ax)
    label_nodes = sorted(connected, key=connected.degree, reverse=True)[:8]
    labels = {node: node for node in label_nodes}
    nx.draw_networkx_labels(
        connected, position, labels=labels, font_size=7, ax=ax,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
    )
    threshold = float(np.min(np.abs(edges[rho_column])))
    ax.set_title(f"Q2 compound coexpression graph (|Spearman rho| >= {threshold:.1f})", loc="left")
    sign_note = "orange: positive; blue: negative" if negative else "all retained edges are positive"
    ax.text(0.01, 0.01,
            f"211 nodes; {len(edges)} edges; {211 - len(connected)} isolates\n"
            f"{sign_note}; node size: degree",
            transform=ax.transAxes, fontsize=9, va="bottom")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(result_dir / "q2_network.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stability(result_dir: Path) -> None:
    top = pd.read_csv(result_dir / "sensory_top3_enriched.csv")
    edges = pd.read_csv(result_dir / "edges.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].hist(top.bootstrap_top3_frequency, bins=np.linspace(0, 1, 21),
                 color="#4472C4", edgecolor="white")
    axes[0].axvline(0.6, color="#C00000", linestyle="--", linewidth=1.5,
                    label="0.6 reference")
    axes[0].axvline(top.bootstrap_top3_frequency.median(), color="#222222",
                    linestyle=":", linewidth=1.5,
                    label=f"median = {top.bootstrap_top3_frequency.median():.3f}")
    axes[0].set(xlabel="Bootstrap Top-3 frequency", ylabel="Compound-sensory rows",
                title="Top-3 selection stability")
    axes[0].legend(frameon=False)

    values = edges.bootstrap_edge_frequency
    axes[1].hist(values, bins=np.linspace(0, 1, 21), color="#70AD47", edgecolor="white")
    axes[1].axvline(0.8, color="#C00000", linestyle="--", linewidth=1.5,
                    label="0.8 reference")
    axes[1].axvline(values.median(), color="#222222", linestyle=":", linewidth=1.5,
                    label=f"median = {values.median():.3f}")
    axes[1].set(xlabel="Bootstrap edge frequency", ylabel="Network edges",
                title="Coexpression-edge stability")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(result_dir / "q2_stability_and_agreement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_top3_heatmap(result_dir: Path) -> None:
    top = pd.read_csv(result_dir / "sensory_top3_enriched.csv").sort_values(["label", "rank"])
    labels = top.label.drop_duplicates().tolist()
    frequency = top.pivot(index="label", columns="rank", values="bootstrap_top3_frequency").loc[labels]
    names = top.pivot(index="label", columns="rank", values="feature").loc[labels]
    fig, ax = plt.subplots(figsize=(10.5, 16))
    image = ax.imshow(frequency.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(labels)):
        for j in range(3):
            feature = str(names.iloc[i, j])
            short = feature if len(feature) <= 18 else feature[:16] + "..."
            ax.text(j, i, f"{short}\n{frequency.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.4, color="white" if frequency.iloc[i, j] >= 0.55 else "#222222")
    ax.set_xticks(range(3), ["Rank 1", "Rank 2", "Rank 3"])
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_title("Top-3 compounds and bootstrap selection frequency", loc="left")
    fig.colorbar(image, ax=ax, label="Bootstrap Top-3 frequency", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(result_dir / "q2_top3_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    result_dir = args.result_dir or args.root / "reports/modeling/q2_final_v1"
    save_network(result_dir)
    save_stability(result_dir)
    save_top3_heatmap(result_dir)
    print(result_dir)


if __name__ == "__main__":
    main()
