#Reproducible Exploratory Data Analysis (EDA)

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# run_eda.py lives in <repo>/src/data/, so parents[2] is <repo>
ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
EDA_REPORT = REPORTS / "05_eda_results.txt"

AI4I_TARGET = "Machine failure"
RT_TARGET = "Attack_type"
RT_BINARY_TARGET = "binary_target"


def ensure_directories() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def savefig(filename: str) -> None:
    path = FIGURES / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def ai4i_eda(lines: list[str]) -> None:
    lines.extend([
        "",
        "=" * 90,
        "AI4I 2020 - TRAINING-SPLIT EDA",
        "=" * 90,
    ])

    df = pd.read_csv(PROCESSED / "ai4i_2020_modeling.csv")
    train = df.loc[df["split"] == "train"].copy()

    lines.append(f"Training rows: {len(train):,}")
    lines.append("Features: 6")
    lines.append(f"Failure rate: {train[AI4I_TARGET].mean() * 100:.3f}%")

    failure_counts = (
        train[AI4I_TARGET]
        .value_counts()
        .sort_index()
        .rename_axis(AI4I_TARGET)
        .reset_index(name="count")
    )
    failure_counts["percentage"] = failure_counts["count"] / len(train) * 100
    failure_counts.to_csv(TABLES / "ai4i_failure_distribution.csv", index=False)

    counts = [
        int(
            failure_counts.loc[
                failure_counts[AI4I_TARGET] == 0, "count"
            ].iloc[0]
        ),
        int(
            failure_counts.loc[
                failure_counts[AI4I_TARGET] == 1, "count"
            ].iloc[0]
        ),
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(["No Failure", "Failure"], counts)
    plt.title("AI4I Machine Failure Distribution - Training Set")
    plt.ylabel("Rows")
    savefig("ai4i_failure_distribution.png")

    by_type = (
        train.groupby("Type")[AI4I_TARGET]
        .agg(rows="size", failures="sum", failure_rate="mean")
        .sort_values("failure_rate", ascending=False)
        .reset_index()
    )
    by_type["failure_rate_pct"] = by_type["failure_rate"] * 100
    by_type.to_csv(TABLES / "ai4i_failure_by_machine_type.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(by_type["Type"], by_type["failure_rate_pct"])
    plt.title("AI4I Failure Rate by Machine Type")
    plt.ylabel("Failure rate (%)")
    plt.xlabel("Machine type")
    savefig("ai4i_failure_by_machine_type.png")

    numeric_features = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    stats = (
        train.groupby(AI4I_TARGET)[numeric_features]
        .agg(["mean", "median", "std"])
        .round(4)
    )
    stats.to_csv(TABLES / "ai4i_feature_statistics_by_failure.csv")

    for feature in numeric_features:
        plt.figure(figsize=(7, 4))
        no_failure = train.loc[train[AI4I_TARGET] == 0, feature].dropna()
        failure = train.loc[train[AI4I_TARGET] == 1, feature].dropna()

        plt.hist(no_failure, bins=30, alpha=0.55, label="No Failure")
        plt.hist(failure, bins=30, alpha=0.55, label="Failure")
        plt.title(f"{feature}: Failure vs No Failure")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()

        safe_name = feature.lower().replace(" ", "_")
        savefig(f"ai4i_{safe_name}_by_failure.png")

    corr_cols = numeric_features + [AI4I_TARGET]
    corr = train[corr_cols].corr()
    corr.to_csv(TABLES / "ai4i_correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=70)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title("AI4I Training-Set Correlation Matrix")
    savefig("ai4i_correlation_matrix.png")

    failure_corr = (
        corr[AI4I_TARGET]
        .drop(AI4I_TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    failure_corr.to_csv(
        TABLES / "ai4i_target_correlations.csv",
        header=["correlation"],
    )

    lines.append("")
    lines.append("AI4I target correlations with numeric features:")
    for feature, value in failure_corr.items():
        lines.append(f"  {feature}: {value:.4f}")

    lines.append("")
    lines.append("AI4I failure rate by machine type:")
    for _, row in by_type.iterrows():
        lines.append(
            f"  {row['Type']}: "
            f"{row['failure_rate_pct']:.3f}% "
            f"({int(row['failures'])}/{int(row['rows'])})"
        )


def rt_iot_eda(lines: list[str]) -> None:
    lines.extend([
        "",
        "=" * 90,
        "RT-IoT2022 - TRAINING-SPLIT EDA",
        "=" * 90,
    ])

    df = pd.read_csv(PROCESSED / "rt_iot2022_modeling.csv")
    train = df.loc[df["split"] == "train"].copy()

    lines.append(f"Training rows: {len(train):,}")
    lines.append(f"Processed columns: {train.shape[1]:,}")

    attack_rate = (train[RT_BINARY_TARGET] == "Attack").mean() * 100
    normal_rate = (train[RT_BINARY_TARGET] == "Normal").mean() * 100
    lines.append(f"Attack rate: {attack_rate:.3f}%")
    lines.append(f"Normal rate: {normal_rate:.3f}%")

    binary_counts = (
        train[RT_BINARY_TARGET]
        .value_counts()
        .rename_axis(RT_BINARY_TARGET)
        .reset_index(name="count")
    )
    binary_counts["percentage"] = binary_counts["count"] / len(train) * 100
    binary_counts.to_csv(
        TABLES / "rt_iot2022_binary_distribution.csv",
        index=False,
    )

    plt.figure(figsize=(6, 4))
    plt.bar(binary_counts[RT_BINARY_TARGET], binary_counts["count"])
    plt.title("RT-IoT2022 Normal vs Attack - Training Set")
    plt.ylabel("Rows")
    savefig("rt_iot2022_binary_distribution.png")

    attack_counts = (
        train[RT_TARGET]
        .value_counts()
        .rename_axis(RT_TARGET)
        .reset_index(name="count")
    )
    attack_counts["percentage"] = attack_counts["count"] / len(train) * 100
    attack_counts.to_csv(
        TABLES / "rt_iot2022_attack_distribution.csv",
        index=False,
    )

    plot_attack = attack_counts.head(12).sort_values("count")

    plt.figure(figsize=(9, 6))
    plt.barh(plot_attack[RT_TARGET], plot_attack["count"])
    plt.title("RT-IoT2022 Traffic Class Distribution")
    plt.xlabel("Rows")
    savefig("rt_iot2022_attack_distribution.png")

    if "proto" in train.columns:
        protocol_counts = (
            train["proto"]
            .value_counts()
            .rename_axis("proto")
            .reset_index(name="count")
        )
        protocol_counts["percentage"] = (
            protocol_counts["count"] / len(train) * 100
        )
        protocol_counts.to_csv(
            TABLES / "rt_iot2022_protocol_distribution.csv",
            index=False,
        )

        plot_protocol = protocol_counts.head(10).sort_values("count")

        plt.figure(figsize=(7, 4))
        plt.barh(plot_protocol["proto"].astype(str), plot_protocol["count"])
        plt.title("RT-IoT2022 Top Protocols")
        plt.xlabel("Rows")
        savefig("rt_iot2022_protocol_distribution.png")

    if "service" in train.columns:
        service_counts = (
            train["service"]
            .value_counts()
            .head(10)
            .rename_axis("service")
            .reset_index(name="count")
        )
        service_counts["percentage"] = (
            service_counts["count"] / len(train) * 100
        )
        service_counts.to_csv(
            TABLES / "rt_iot2022_service_distribution.csv",
            index=False,
        )

        plot_services = service_counts.sort_values("count")

        plt.figure(figsize=(7, 5))
        plt.barh(plot_services["service"].astype(str), plot_services["count"])
        plt.title("RT-IoT2022 Top Services")
        plt.xlabel("Rows")
        savefig("rt_iot2022_service_distribution.png")

    exclude_cols = {
        "row_id",
        "duplicate_group_id",
        "duplicate_group_size",
    }

    numeric_cols = [
        c
        for c in train.select_dtypes(include=np.number).columns
        if c not in exclude_cols
    ]

    numeric_summary = train[numeric_cols].describe().T.sort_index()
    numeric_summary.to_csv(
        TABLES / "rt_iot2022_numeric_feature_summary.csv"
    )

    lines.append(
        f"Numeric features observed: {len(numeric_cols):,}"
    )

    encoded_target = (train[RT_BINARY_TARGET] == "Attack").astype(int)

    corr_series = (
        train[numeric_cols]
        .corrwith(encoded_target)
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    corr_series.head(20).to_csv(
        TABLES / "rt_iot2022_top_numeric_target_correlations.csv",
        header=["correlation"],
    )

    top_corr = corr_series.head(15).sort_values()

    plt.figure(figsize=(8, 6))
    plt.barh(top_corr.index.astype(str), top_corr.values)
    plt.title(
        "RT-IoT2022 Top Numeric Feature Correlations "
        "with Attack Target"
    )
    plt.xlabel("Correlation with Attack=1")
    savefig("rt_iot2022_top_numeric_target_correlations.png")

    lines.append("")
    lines.append(
        "Top numeric feature correlations with binary Attack target:"
    )
    for feature, value in corr_series.head(15).items():
        lines.append(f"  {feature}: {value:.4f}")

    lines.append("")
    lines.append("RT-IoT2022 attack-type distribution:")
    for _, row in attack_counts.iterrows():
        lines.append(
            f"  {row[RT_TARGET]}: "
            f"{int(row['count']):,} "
            f"({row['percentage']:.3f}%)"
        )


def write_report(lines: list[str]) -> None:
    EDA_REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()

    lines = [
        "SECUREPREDICT V0.1 - EXPLORATORY DATA ANALYSIS",
        "=" * 90,
        "EDA is performed on the TRAIN split only.",
        "Validation and test data remain untouched for model selection.",
        "",
    ]

    print(lines[0])
    print(lines[1])
    print("Using leakage-safe processed datasets.")
    print("Generating AI4I EDA...")

    ai4i_eda(lines)
    print("AI4I EDA complete.")

    print("Generating RT-IoT2022 EDA...")
    rt_iot_eda(lines)
    print("RT-IoT2022 EDA complete.")

    lines.extend([
        "",
        "=" * 90,
        "EDA COMPLETE",
        "=" * 90,
        f"Report: {EDA_REPORT}",
        f"Figures: {FIGURES}",
        f"Tables: {TABLES}",
    ])

    write_report(lines)

    print("")
    print("=" * 90)
    print("EDA COMPLETE")
    print("=" * 90)
    print(f"Report:  {EDA_REPORT}")
    print(f"Figures: {FIGURES}")
    print(f"Tables:  {TABLES}")


if __name__ == "__main__":
    main()
