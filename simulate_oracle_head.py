# simulate_oracle_2.py

import pandas as pd

def load_expression_preview(input_path: str, n_rows: int = 10):
    """
    Load and preview gene expression matrix from .tsv.gz file.

    Parameters:
        input_path (str): Path to the input .tsv.gz file.
        n_rows (int): Number of rows to preview.

    Returns:
        pd.DataFrame: Preview of the expression matrix.
    """
    try:
        df = pd.read_csv(input_path, sep="\t", index_col=0)
        df = df[~df["gene_name"].duplicated()]
        df.index = df["gene_name"]
        df = df.drop(columns=["gene_name"])
        preview = df.head(n_rows)
        return preview
    except Exception as e:
        print(f"❌ Error loading preview: {e}")
        return None
