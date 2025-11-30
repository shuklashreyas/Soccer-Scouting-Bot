# src/modeling/show_clusters.py

import pandas as pd
from src.modeling.similarity import PlayerEmbeddingModel, FRIENDLY_NAMES

DATA_PATH = "data/processed/all_leagues_clean.csv"


def summarize_cluster(df, feature_cols, cluster_id, top_n=10):
    """Return representative players + mean stats for a cluster."""
    sub = df[df["role_cluster"] == cluster_id]

    # Core shown columns
    cols = ["Player"]
    if "Squad" in sub.columns: cols.append("Squad")
    if "Pos" in sub.columns: cols.append("Pos")
    reps = sub[cols].head(top_n)

    means = sub[feature_cols].mean().sort_values(ascending=False)

    pretty = {}
    for stat, value in means.items():
        name = FRIENDLY_NAMES.get(stat, stat)
        pretty[name] = round(value, 3)

    return reps, pretty


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n==============================")
    print("   ROLE CLUSTERS DISCOVERED")
    print("==============================\n")

    # 🔥 Compute embeddings + clusters
    model = PlayerEmbeddingModel(df)
    df = model.df  # this now contains df["role_cluster"]

    # Get clusters that actually exist
    all_clusters = sorted(df["role_cluster"].unique().tolist())
    feature_cols = model.feature_cols

    for cid in all_clusters:
        print("\n==========================================")
        print(f"   CLUSTER {cid}")
        print("==========================================")

        reps, means = summarize_cluster(df, feature_cols, cid)

        print("\nRepresentative Players:")
        print(reps.to_string(index=False))

        print("\nTop Stats (cluster means):")
        for stat, val in list(means.items())[:12]:
            print(f"- {stat}: {val}")

        print("\n------------------------------------------")


if __name__ == "__main__":
    main()
