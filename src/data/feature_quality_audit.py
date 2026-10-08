#Feature Quality Audit


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"

AI4I_TARGET = "Machine failure"
RT_TARGET = "Attack_type"
RT_BINARY_TARGET = "binary_target"

# Metadata
RT_METADATA_COLUMNS = {
    "row_id",
    "duplicate_group_id",
    "duplicate_group_size",
    "split",
    RT_TARGET,
    RT_BINARY_TARGET,
}

AI4I_METADATA_COLUMNS = {
    "row_id",
    "split",
    AI4I_TARGET,
}


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLES / filename
    df.to_csv(path, index=False)
    return path


def audit_ai4i(report: list[str]) -> None:
    report.extend([
        "",
        "=" * 90,
        "AI4I 2020 - FEATURE QUALITY AUDIT",
        "=" * 90,
    ])

    df = pd.read_csv(PROCESSED / "ai4i_2020_modeling.csv")
    train = df[df["split"] == "train"].copy()

    features = [
        c for c in train.columns
        if c not in AI4I_METADATA_COLUMNS
    ]

    summary = []

    for col in features:
        s = train[col]
        value_counts = s.value_counts(dropna=False)

        summary.append({
            "feature": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=False)),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "most_common_value": (
                value_counts.index[0]
                if len(value_counts)
                else None
            ),
            "most_common_count": (
                int(value_counts.iloc[0])
                if len(value_counts)
                else 0
            ),
            "dominant_pct": (
                float(value_counts.iloc[0] / len(s) * 100)
                if len(value_counts)
                else 0.0
            ),
            "is_constant": bool(s.nunique(dropna=False) <= 1),
            "is_near_constant": bool(
                len(value_counts) > 0
                and value_counts.iloc[0] / len(s) >= 0.995
            ),
        })

    summary_df = pd.DataFrame(summary)
    save_table(summary_df, "ai4i_feature_quality_summary.csv")

    constants = summary_df[summary_df["is_constant"]].copy()
    near_constants = summary_df[
        summary_df["is_near_constant"]
        & ~summary_df["is_constant"]
    ].copy()

    report.append(
        f"Training rows: {len(train):,}"
    )
    report.append(
        f"Candidate features: {len(features)}"
    )
    report.append(
        f"Constant features: {len(constants)}"
    )
    report.append(
        f"Near-constant features (>=99.5% one value): "
        f"{len(near_constants)}"
    )

    if len(constants):
        report.append(
            "Constant features: "
            + ", ".join(constants["feature"].astype(str))
        )
    else:
        report.append("Constant features: None")

    if len(near_constants):
        report.append(
            "Near-constant features: "
            + ", ".join(near_constants["feature"].astype(str))
        )
    else:
        report.append("Near-constant features: None")

    report.append("")


def audit_rt_iot(report: list[str]) -> None:
    report.extend([
        "",
        "=" * 90,
        "RT-IoT2022 - FEATURE QUALITY AUDIT",
        "=" * 90,
    ])

    df = pd.read_csv(PROCESSED / "rt_iot2022_modeling.csv")
    train = df[df["split"] == "train"].copy()

    feature_columns = [
        c for c in train.columns
        if c not in RT_METADATA_COLUMNS
    ]

    numeric_features = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(train[c])
    ]

    categorical_features = [
        c for c in feature_columns
        if c not in numeric_features
    ]

    # Numeric audit
    numeric_rows = []

    for col in numeric_features:
        s = train[col]
        counts = s.value_counts(dropna=False)

        numeric_rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=False)),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "dominant_pct": float(counts.iloc[0] / len(s) * 100),
            "is_constant": bool(s.nunique(dropna=False) <= 1),
            "is_near_constant": bool(
                counts.iloc[0] / len(s) >= 0.995
            ),
        })

    numeric_summary = pd.DataFrame(numeric_rows)
    save_table(
        numeric_summary,
        "rt_iot2022_numeric_feature_quality.csv",
    )

    constants = numeric_summary[
        numeric_summary["is_constant"]
    ].copy()

    near_constants = numeric_summary[
        numeric_summary["is_near_constant"]
        & ~numeric_summary["is_constant"]
    ].copy()

    # Categorical audit
    categorical_rows = []

    for col in categorical_features:
        s = train[col]
        counts = s.value_counts(dropna=False)
        unique_ratio = s.nunique(dropna=False) / len(s)

        categorical_rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=False)),
            "unique_ratio": float(unique_ratio),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "most_common_value": (
                str(counts.index[0]) if len(counts) else None
            ),
            "most_common_count": (
                int(counts.iloc[0]) if len(counts) else 0
            ),
            "dominant_pct": (
                float(counts.iloc[0] / len(s) * 100)
                if len(counts) else 0.0
            ),
        })

    categorical_summary = pd.DataFrame(categorical_rows)
    save_table(
        categorical_summary,
        "rt_iot2022_categorical_feature_quality.csv",
    )

    high_cardinality = categorical_summary[
        (categorical_summary["unique_ratio"] >= 0.05)
        | (categorical_summary["n_unique"] >= 100)
    ].copy()

    report.append(
        f"Training rows: {len(train):,}"
    )
    report.append(
        f"Candidate feature columns: {len(feature_columns)}"
    )
    report.append(
        f"Numeric features: {len(numeric_features)}"
    )
    report.append(
        f"Categorical features: {len(categorical_features)}"
    )
    report.append(
        f"Constant numeric features: {len(constants)}"
    )
    report.append(
        f"Near-constant numeric features (>=99.5% one value): "
        f"{len(near_constants)}"
    )

    if len(constants):
        report.append(
            "Constant numeric features: "
            + ", ".join(constants["feature"].astype(str))
        )
    else:
        report.append("Constant numeric features: None")

    if len(near_constants):
        report.append(
            "Near-constant numeric features: "
            + ", ".join(near_constants["feature"].astype(str))
        )
    else:
        report.append("Near-constant numeric features: None")

    report.append("")
    report.append("Categorical feature cardinality:")

    if len(categorical_summary):
        for _, row in categorical_summary.sort_values(
            "n_unique",
            ascending=False,
        ).iterrows():
            report.append(
                f"  {row['feature']}: "
                f"{int(row['n_unique']):,} unique "
                f"({row['unique_ratio'] * 100:.3f}% of rows)"
            )
    else:
        report.append("  None")

    report.append("")
    report.append("High-cardinality categorical candidates for review:")

    if len(high_cardinality):
        for _, row in high_cardinality.iterrows():
            report.append(
                f"  {row['feature']}: "
                f"{int(row['n_unique']):,} unique"
            )
    else:
        report.append("  None")

    #ID-like numeric candidates: not automatic deletions.For human review only. These are numeric features that have a high number of unique values, which may indicate they are identifiers rather than useful features for modeling.
    id_like = numeric_summary[
        (
            numeric_summary["n_unique"]
            / len(train)
        ) >= 0.90
    ].copy()

    save_table(
        id_like,
        "rt_iot2022_id_like_numeric_candidates.csv",
    )

    report.append("")
    report.append(
        "High-uniqueness numeric candidates for review "
        "(>=90% unique):"
    )

    if len(id_like):
        for _, row in id_like.iterrows():
            report.append(
                f"  {row['feature']}: "
                f"{int(row['n_unique']):,} unique "
                f"({row['n_unique'] / len(train) * 100:.2f}%)"
            )
    else:
        report.append("  None")

    report.append("")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    report: list[str] = [
        "SECUREPREDICT V0.1.1 - FEATURE QUALITY AUDIT",
        "=" * 90,
        "Audit performed on TRAIN splits only.",
        "No modeling datasets are modified by this script.",
    ]

    print(report[0])
    print(report[1])
    print("Auditing AI4I feature quality...")
    audit_ai4i(report)
    print("AI4I feature quality audit complete.")

    print("Auditing RT-IoT2022 feature quality...")
    audit_rt_iot(report)
    print("RT-IoT2022 feature quality audit complete.")

    report.extend([
        "",
        "=" * 90,
        "FEATURE QUALITY AUDIT COMPLETE",
        "=" * 90,
        f"Tables: {TABLES}",
    ])

    path = REPORTS / "06_feature_quality_audit.txt"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("")
    print("=" * 90)
    print("FEATURE QUALITY AUDIT COMPLETE")
    print("=" * 90)
    print(f"Report: {path}")
    print(f"Tables: {TABLES}")


if __name__ == "__main__":
    main()
