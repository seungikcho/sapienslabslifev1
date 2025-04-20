# analyze_oracle.py

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import networkx as nx
import os
from fpdf import FPDF
from PIL import Image

def load_oracle(path="/tmp/oracle_object.pkl"):
    with open(path, "rb") as f:
        oracle = pickle.load(f)
    return oracle


def plot_deltaX_umap(oracle, save_path="/tmp/umap_deltaX.png"):
    delta_X = oracle.adata.layers["simulated_count"] - oracle.adata.X
    delta_magnitude = np.linalg.norm(delta_X, axis=1)
    oracle.adata.obs["delta_magnitude"] = delta_magnitude
    sc.pl.umap(oracle.adata, color="delta_magnitude", cmap="magma", size=50, show=False)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cluster_barplot(oracle, save_path="/tmp/cluster_deltaX_bar.png"):
    cluster_mean = oracle.adata.obs.groupby("cluster")["delta_magnitude"].mean()
    cluster_mean.plot(kind="bar", color="salmon", title="Avg ΔX by Cluster")
    plt.ylabel("Expression Shift Magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cluster_deg(oracle, save_path="/tmp/cluster_DEG_top10.png"):
    sc.tl.rank_genes_groups(oracle.adata, groupby="cluster", method="wilcoxon")
    sc.pl.rank_genes_groups(oracle.adata, n_genes=10, sharey=False, show=False)
    plt.savefig(save_path, dpi=300)
    plt.close()


def export_top_delta_genes(oracle, n=20, save_path="/tmp/top_deltaX_genes.csv"):
    delta_X = oracle.adata.layers["simulated_count"] - oracle.adata.X
    gene_delta = np.mean(np.abs(delta_X), axis=0)
    top_delta_genes = pd.Series(gene_delta, index=oracle.adata.var_names).sort_values(ascending=False)
    top_20 = top_delta_genes.head(n)
    top_20.to_csv(save_path)
    return top_20

def plot_tf_target_network(corr_df, save_path="/tmp/tf_target_network.png", top_n=10):

    df = corr_df.sort_values(by="corr", ascending=False).head(top_n)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["TF"], row["target"], weight=row["corr"])

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, arrows=True, width=weights, node_size=1500,
            node_color="skyblue", edge_color="gray", font_size=10)
    plt.title("Top TF → Target Gene Relationships")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_report():
    image_paths = {
    "Quiver Plot": "/tmp/quiver_result.png",
    "UMAP DX": "/tmp/umap_deltaX.png",           
    "Cluster DX Bar": "/tmp/cluster_deltaX_bar.png",
    "Cluster DEGs": "/tmp/cluster_DEG_top10.png",
    "TF-Target Network": "/tmp/tf_target_network.png"
    }

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for title, path in image_paths.items():
        if os.path.exists(path):
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            safe_title = title.encode("latin-1", "ignore").decode("latin-1")
            pdf.cell(0, 10, safe_title, ln=True)

            img = Image.open(path).convert("RGB")
            temp_path = f"/tmp/temp_{title}.jpg"
            img.save(temp_path)
            pdf.image(temp_path, x=10, y=30, w=180)
            os.remove(temp_path)

    report_path = "/tmp/sapiens_report.pdf"
    pdf.output(report_path)
    return report_path