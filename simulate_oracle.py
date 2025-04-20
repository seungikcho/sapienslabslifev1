# simulate_oracle.py - local use

import pandas as pd
import numpy as np
import anndata
import celloracle as co
import matplotlib.pyplot as plt
import scanpy as sc


def run_oracle_simulation(input_path: str, output_path: str, perturb_dict: dict):
    try:
        # 1. Load expression matrix (.tsv.gz)
        counts_df = pd.read_csv(input_path, sep="\t", index_col=0)
        counts_df = counts_df[~counts_df["gene_name"].duplicated()]
        counts_df.index = counts_df["gene_name"]
        counts_df = counts_df.drop(columns=["gene_name"])

        # 2. Preprocess
        expr_matrix_log = np.log1p(counts_df)
        adata = anndata.AnnData(X=expr_matrix_log.T)
        adata.var_names = expr_matrix_log.index
        adata.obs_names = expr_matrix_log.columns
        adata.obs["cluster"] = np.random.choice(["C1", "C2", "C3"], size=adata.n_obs)
        adata.obsm["X_umap"] = np.random.rand(adata.n_obs, 2)

        # 3. Initialize Oracle
        oracle = co.Oracle()
        oracle.import_anndata_as_raw_count(
            adata, cluster_column_name="cluster", embedding_name="X_umap", transform="log1p"
        )
        oracle.perform_PCA()
        oracle.knn_imputation()

        # 4. TF dictionary (example - will update further on dictionary)
        tf_genes = ["fnr", "resD", "sigB", "nsrR"]
        target_genes = [g for g in adata.var_names if g not in tf_genes]
        TF_dict = {tf: target_genes for tf in tf_genes}
        oracle.import_TF_data(TFdict=TF_dict)

        # 5. GRN inference
        oracle.get_links(cluster_name_for_GRN_unit="cluster", alpha=1, bagging_number=10, model_method="bagging_ridge")
        oracle.fit_GRN_for_simulation(GRN_unit="cluster")

        # 6. Perturbation simulation
        print(f"🔧 Running simulation with perturbation: {perturb_dict}")
        oracle.simulate_shift(
            perturb_condition=perturb_dict,
            GRN_unit="cluster",
            n_propagation=3,
            clip_delta_X=True
        )
        oracle.estimate_transition_prob(n_neighbors=30, knn_random=True, sampled_fraction=1)
        oracle.calculate_embedding_shift(sigma_corr=0.05)

        # 7. Save quiver plot
        oracle.plot_quiver(scale=20)
        fig = plt.gcf()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print("✅ Simulation finished. Quiver plot saved to output path.")

        delta_X = oracle.adata.layers["simulated_count"] - oracle.adata.X
        delta_magnitude = np.linalg.norm(delta_X, axis=1)
        oracle.adata.obs["delta_magnitude"] = delta_magnitude

        # Save Oracle object for downstream use
        import pickle
        with open("/tmp/oracle_object.pkl", "wb") as f:
            pickle.dump(oracle, f)

        print("📦 Oracle object saved to /tmp/oracle_object.pkl for later analysis.")

    except Exception as e:
        print(f"❌ Error: {e}")
