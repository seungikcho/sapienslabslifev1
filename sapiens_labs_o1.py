import streamlit as st
import re
import os
import numpy as np
import uuid
import pandas as pd
from simulate_oracle import run_oracle_simulation
from simulate_oracle_head import load_expression_preview
from analyze_oracle import load_oracle, plot_deltaX_umap, plot_cluster_barplot, plot_cluster_deg, export_top_delta_genes
from openai import OpenAI

# OpenAI API Key 
client = OpenAI(api_key="private--")

st.set_page_config(page_title="Sapiens Labs", layout="wide", page_icon="🧬")
# Roboto font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, input, button {
        font-family: 'Roboto', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

if "step" not in st.session_state:
    st.session_state.step = 0 
if "input_path" not in st.session_state:
    st.session_state.input_path = None
if "suggested_genes" not in st.session_state:
    st.session_state.suggested_genes = []
if "perturb_dict" not in st.session_state:
    st.session_state.perturb_dict = {}

if st.session_state.step == 0:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 85vh;">
            <h1 style="font-size: 3.5em; font-weight: 800; font-family: 'Roboto', sans-serif;">🧬 Sapiens Labs</h1>
            <p style="font-size: 1.2em; color: gray; margin-top: -10px;">Reprogramming microbes to fix the planet</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button("🚀 Start", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# STEP 1: Upload RNA-seq file
elif st.session_state.step == 1:
    st.header("Upload your RNA-seq gene count matrix (.tsv.gz)")
    uploaded_file = st.file_uploader("Upload your .tsv.gz file", type=["tsv", "gz"])

    if uploaded_file:
        with open("temp_input.tsv.gz", "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.input_path = "temp_input.tsv.gz"

        if st.button("➡️ Next"):
            st.session_state.step = 2
            st.rerun()

# STEP 2: Preview uploaded data
elif st.session_state.step == 2:
    st.header("Preview Expression Matrix")

    preview_df = load_expression_preview(st.session_state.input_path)
    if preview_df is not None:
        st.dataframe(preview_df)
        st.session_state.preview_df = preview_df
    else:
        st.error("❌ Failed to preview data.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("➡️ Next (Prompt)"):
            st.session_state.step = 3
            st.rerun()

# STEP 3: Prompt input and gene suggestion
elif st.session_state.step == 3:
    st.header("Describe your goal to save Earth")

    prompt = st.text_area("🔬 What do you want to optimize?",
                          placeholder="e.g., I want to improve nitrate decomposition in Bacillus subtilis.")

    if st.button("🔍 Analyze Prompt"):
        if not prompt.strip():
            st.warning("Please enter a biological goal.")
        else:
            with st.spinner("🔍 Analyzing with Sapiens AI..."):
                try:
                    gene_context = "\n".join(st.session_state.preview_df.index[:3000])
                    query = f"""
You're a microbial genomics assistant. Below are genes detected in RNA-seq:
{gene_context}

Given the goal: {prompt}
Which genes should be perturbed (overexpressed or knocked-in) to achieve this? Return only gene names found in the provided list.
"""
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for gene perturbation in RNA-seq analysis."},
                            {"role": "user", "content": query}
                        ],
                        temperature=0.4
                    )

                    gpt_reply = response.choices[0].message.content
                    st.markdown("### Sapiens AI Response:")
                    st.markdown(gpt_reply)

                    candidate_genes = re.findall(r'\b[a-zA-Z]{2,10}[0-9]*\b', gpt_reply)
                    candidate_genes = list(set(candidate_genes))
                    df = st.session_state.preview_df
                    genes_found = [g for g in candidate_genes if g in df.index]

                    if genes_found:
                        st.session_state.suggested_genes = genes_found
                        st.session_state.perturb_dict = {g: 2.0 for g in genes_found}
                        st.success(f"✅ Found {len(genes_found)} genes in your dataset: {', '.join(genes_found)}")
                        st.session_state.step = 4
                        st.rerun()
                    else:
                        st.error("❌ No genes from GPT response matched your dataset.")

                except Exception as e:
                    st.error(f"OpenAI API Error: {e}")

    if st.button("⬅️ Back"):
        st.session_state.step = 2
        st.rerun()

# STEP 4: Set perturbation values and run simulation
elif st.session_state.step == 4:
    st.header("Set Perturbation Values")

    st.markdown("Adjust overexpression values for the suggested genes below:")
    for gene in st.session_state.suggested_genes:
        unique_key = f"perturb_input_{gene}_{uuid.uuid4()}"  # 항상 다른 key 보장
        val = st.number_input(
            f"{gene} expression level",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.perturb_dict.get(gene, 2.0),
            step=0.1,
            key=unique_key
        )
        st.session_state.perturb_dict[gene] = val

    if st.button("🚀 Run simulate_shift"):
        st.info("Running simulation...")
        output_path = "/tmp/quiver_result.png"

        run_oracle_simulation(
            input_path=st.session_state.input_path,
            output_path=output_path,
            perturb_dict=st.session_state.perturb_dict
        )

        st.session_state.step = 5
        st.rerun()

    if st.button("⬅️ Back"):
        st.session_state.step = 3
        st.rerun()

# STEP 5: UMAP & Top Gene Analysis
elif st.session_state.step == 5:
    st.header("UMAP & Top Gene Analysis")
    oracle = load_oracle()

    if st.button("🧬 Run Analysis & Show Results"):
        st.info("Running analysis...")
        result_success = False

        delta_umap = plot_deltaX_umap(oracle)
        cluster_bar = plot_cluster_barplot(oracle)
        cluster_deg = plot_cluster_deg(oracle)
        top20 = export_top_delta_genes(oracle)

        if top20 is not None:
            st.session_state.top_genes = top20
            result_success = True

        if result_success:
            st.success("✅ Analysis complete!")
            st.rerun()
        else:
            st.error("❌ Analysis incomplete. Check your input data and outputs.")

    if os.path.exists("/tmp/umap_deltaX.png"):
        st.image("/tmp/umap_deltaX.png", caption="ΔX by Cell", use_container_width=True)
    if os.path.exists("/tmp/cluster_deltaX_bar.png"):
        st.image("/tmp/cluster_deltaX_bar.png", caption="Avg ΔX by Cluster", use_container_width=True)
    if os.path.exists("/tmp/cluster_DEG_top10.png"):
        st.image("/tmp/cluster_DEG_top10.png", caption="Top DEGs per Cluster", use_container_width=True)

    if "top_genes" in st.session_state:
        st.subheader("Top ΔX Genes")
        st.dataframe(st.session_state.top_genes)

    if st.button("➡️ Next: TF Correlation Analysis"):
        st.session_state.step = 6
        st.rerun()

# Step 6: TF-target network analysis
elif st.session_state.step == 6:
    st.header("TF ↔ Target Network Analysis")

    st.markdown("Top TFs inferred by ΔX correlation with target genes.")
    oracle = load_oracle()

    delta_X = oracle.adata.layers["simulated_count"] - oracle.adata.X
    delta_X_np = delta_X

    tf_genes = ["fnr", "resD", "sigB", "nsrR"]
    key_genes = st.session_state.top_genes.index.tolist()

    correlation_results = []
    for tf in tf_genes:
        if tf in oracle.adata.var_names:
            tf_idx = oracle.adata.var_names.get_loc(tf)
            tf_delta = delta_X_np[:, tf_idx]
            for gene in key_genes:
                if gene in oracle.adata.var_names:
                    gene_idx = oracle.adata.var_names.get_loc(gene)
                    gene_delta = delta_X_np[:, gene_idx]
                    corr = np.corrcoef(tf_delta, gene_delta)[0, 1]
                    correlation_results.append({"TF": tf, "target": gene, "corr": corr})

    corr_df = pd.DataFrame(correlation_results).sort_values(by="corr", ascending=False)

    from analyze_oracle import plot_tf_target_network
    network_path = "/tmp/tf_target_network.png"
    plot_tf_target_network(corr_df, save_path=network_path, top_n=12)

    st.image(network_path, caption="TF ↔ Target Network", use_container_width=True)
    st.dataframe(corr_df.head(10))

    st.session_state.corr_df = corr_df
    st.session_state.tf_importance_df = pd.DataFrame({  # 임시 placeholder
        "TF": tf_genes,
        "Importance": np.random.rand(len(tf_genes))  # 나중에 실제 값으로 대체
    })

    st.subheader("📥 Download Simulation Report")
    from analyze_oracle import generate_report

    report_path = generate_report()
    with open(report_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="sapiens_report.pdf", mime="application/pdf")

    if st.button("⬅️ Back to Step 5"):
        st.session_state.step = 5
        st.rerun()

    if st.button("➡️ Next: AI Analysis & Suggestions"):
        st.session_state.step = 7
        st.rerun()

# STEP 7: AI Interpretation and Gene Combination Suggestion
elif st.session_state.step == 7:
    st.header("AI Interpretation")

    st.markdown("This AI assistant will interpret the simulation results and suggest new gene perturbation combinations based on the analysis.")

    if st.button("🤖 Analyze with Sapiens Labs"):
        with st.spinner("Running Sapiens Labs analysis..."):
            try:
                summary_prompt = f"""
You are a genomics research AI assistant helping to design better perturbation strategies for synthetic biology experiments.

Here is the result of a gene perturbation simulation on a bacterial gene expression dataset:

🧬 Top ΔX Genes (genes with highest change after simulation):
{st.session_state.top_genes.to_string(index=True)}

🔬 TF Importance Ranking (by how strongly each TF drives gene expression shift):
{st.session_state.tf_importance_df.to_string(index=False)}

🔁 TF ↔ Target Correlation (from ΔX propagation):
{st.session_state.corr_df.head(10).to_string(index=False)}

---

Instructions:
1. Based on the data above, explain what the simulation reveals about which genes and TFs are most impactful.
2. Identify any patterns you see (e.g., co-regulation, TFs linked to multiple high ΔX targets).
3. Propose 1–2 new gene perturbation combinations (TFs or genes to overexpress/knock-in) that would likely amplify or stabilize the observed beneficial effects.
4. Justify your recommendations with reasoning.

Use clear, concise language understandable to a synthetic biology researcher.
"""

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a genomics research assistant."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.6
                )

                reply = response.choices[0].message.content
                st.markdown("### 🤖 Sapiens Labs Interpretation & Recommendations:")
                st.markdown(reply)

                st.session_state["GPT_reply_text"] = reply

            except Exception as e:
                st.error(f"GPT API Error: {e}")

    if st.button("🔁 Test This Combo"):
        if "GPT_reply_text" in st.session_state:
            gpt_text = st.session_state["GPT_reply_text"]

            import re
            from analyze_oracle import load_oracle
            oracle = load_oracle()

            gene_candidates = re.findall(r'\b[a-zA-Z]{2,10}[0-9]*\b', gpt_text)
            valid_genes = list(set([g for g in gene_candidates if g in oracle.adata.var_names]))

            if valid_genes:
                st.session_state.suggested_genes = valid_genes
                st.session_state.perturb_dict = {g: 2.0 for g in valid_genes}
                st.success(f"✅ Found {len(valid_genes)} valid genes: {', '.join(valid_genes)}")
                st.session_state.step = 4
                st.rerun()
            else:
                st.error("❌ No valid genes found in GPT output.")
        else:
            st.warning("Please run GPT analysis first.")

    if "GPT_reply_text" in st.session_state:
        if st.button("➡️ Optimize Perturbation Values"):
            st.session_state.step = 8
            st.rerun()

    if st.button("⬅️ Back to Step 6"):
        st.session_state.step = 6
        st.rerun()

# STEP 8: Optimal Perturbation Combination Search
elif st.session_state.step == 8:
    st.header("Perturbation Combination Optimization")

    if "GPT_reply_text" not in st.session_state:
        st.warning("❌ No GPT result found. Please run analysis in Step 7.")
        st.stop()

    st.markdown("Based on Sapiens Labs gene combination suggestion, please select the genes you want to include in the simulation.")

    import re
    from analyze_oracle import load_oracle
    oracle = load_oracle()

    gpt_text = st.session_state["GPT_reply_text"]
    gene_candidates = list(set(re.findall(r'\b[a-zA-Z_]{2,10}[0-9]*\b', gpt_text)))

    valid_genes = [g for g in gene_candidates if g in oracle.adata.var_names]

    if not valid_genes:
        st.error("❌ No valid genes found in RNA-seq data.")
        st.stop()

    selected_genes = []
    st.markdown("🧬 **Genes detected from Sapiens Labs**")
    cols = st.columns(4)
    for i, gene in enumerate(valid_genes):
        with cols[i % 4]:
            if st.checkbox(gene):
                selected_genes.append(gene)

    if not selected_genes:
        st.info("ℹ️ Please select at least one gene to run the simulation.")
        st.stop()

    min_val = st.number_input("Min perturbation value", 0.1, 10.0, value=0.5, step=0.1)
    max_val = st.number_input("Max perturbation value", min_val + 0.1, 10.0, value=3.0, step=0.1)
    step_val = st.number_input("Step size", 0.1, 1.0, value=0.5, step=0.1)

    if st.button("🔁 Run automated simulation"):
        import itertools
        import numpy as np
        import pandas as pd
        from simulate_oracle import run_oracle_simulation
        from analyze_oracle import load_oracle

        with st.spinner("Simulating all combinations... please wait ⏳"):
            value_range = np.round(np.arange(min_val, max_val + step_val, step_val), 2)
            combos = list(itertools.product(value_range, repeat=len(selected_genes)))

            result_table = []
            for idx, combo in enumerate(combos):
                perturb = {gene: val for gene, val in zip(selected_genes, combo)}

                try:
                    run_oracle_simulation(
                        input_path=st.session_state.input_path,
                        output_path=f"/tmp/step8_combo_{idx}.png",
                        perturb_dict=perturb,
                    )
                    oracle = load_oracle()

                    if "delta_magnitude" not in oracle.adata.obs:
                        raise KeyError("delta_magnitude not found in oracle.adata.obs")

                    avg_delta = float(np.mean(oracle.adata.obs["delta_magnitude"]))
                    top_cluster = oracle.adata.obs["cluster"].value_counts().idxmax()

                    result_table.append({
                        "Combo ID": idx + 1,
                        "Perturbed Genes": ", ".join([f"{k}:{v}" for k, v in perturb.items()]),
                        "Avg ΔX": round(avg_delta, 3),
                        "Dominant Cluster": top_cluster,
                    })

                except Exception as e:
                    print(f"❌ Failed combo: {perturb} → {e}")
                    continue

            if result_table:
                df_result = pd.DataFrame(result_table).sort_values(by="Avg ΔX", ascending=False)
                st.session_state.optimization_result = df_result

                best = df_result.iloc[0]
                st.success("✅ Best perturbation combination found:")
                st.markdown(f"**Genes:** `{best['Perturbed Genes']}`")
                st.markdown(f"**Avg ΔX:** `{best['Avg ΔX']}`")
                st.markdown(f"**Dominant Cluster:** `{best['Dominant Cluster']}`")

            else:
                st.error("❌ No successful simulations. Try adjusting the range or gene selection.")

    if st.button("➡️ View All Results in Step 9"):
        st.session_state.step = 9
        st.rerun()

# STEP 9: Summary of Best Perturbation Combinations
elif st.session_state.step == 9:
    st.header("Best Perturbation Combinations Summary")

    if "optimization_result" not in st.session_state:
        st.warning("No optimization results found. Please run Step 8 first.")
    else:
        df_result = st.session_state.optimization_result

        st.markdown("🔝 **Top-performing perturbation combinations** based on Avg ΔX (expression shift):")
        st.dataframe(df_result.head(10), use_container_width=True)

        top_combo = df_result.iloc[0]
        st.markdown("### 🏆 Best Combination")
        st.markdown(f"**Combo ID:** {top_combo['Combo ID']}")
        st.markdown(f"**Genes:** `{top_combo['Perturbed Genes']}`")
        st.markdown(f"**Avg ΔX:** {top_combo['Avg ΔX']}")
        st.markdown(f"**Dominant Cluster:** {top_combo['Dominant Cluster']}")

        if st.button("🔁 Re-run Best Combo Simulation"):
            gene_combo_dict = dict([tuple(g.split(":")) for g in top_combo["Perturbed Genes"].split(", ")])
            perturb_dict = {gene: float(val) for gene, val in gene_combo_dict.items()}

            st.session_state.suggested_genes = list(perturb_dict.keys())
            st.session_state.perturb_dict = perturb_dict
            st.success(f"Running simulation for best combo: {perturb_dict}")
            st.session_state.step = 4
            st.rerun()

        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=df_result.to_csv(index=False),
            file_name="perturbation_optimization_results.csv",
            mime="text/csv"
        )

        if st.button("🔄 Try Another Combination"):
            st.session_state.step = 7
            st.rerun()
