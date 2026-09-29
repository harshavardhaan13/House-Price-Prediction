"""
SmartHouse AI - Exploratory Data Analysis & Real Estate Market Insights Module.
Provides modular statistical analysis, target distribution inspection,
locality valuation benchmarking, correlation profiling, and publication-ready visualizations.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.utils.config import CLEANED_DATA_PATH, VISUALIZATIONS_DIR

logger = logging.getLogger("SmartHouseAI.EDA")

# Set standard styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def compute_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes general shape, column composition, and memory footprint.
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }


def compute_target_statistics(df: pd.DataFrame, target_col: str = "price") -> Dict[str, Any]:
    """
    Computes detailed central tendency, dispersion, and skewness for target variable.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    series = df[target_col].dropna()
    log_series = np.log1p(series[series > 0])

    return {
        "target_name": target_col,
        "unit": "Lakhs INR",
        "count": int(len(series)),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std": round(float(series.std()), 2),
        "min": round(float(series.min()), 2),
        "q25": round(float(series.quantile(0.25)), 2),
        "q75": round(float(series.quantile(0.75)), 2),
        "max": round(float(series.max()), 2),
        "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 2),
        "skewness": round(float(series.skew()), 2),
        "kurtosis": round(float(series.kurtosis()), 2),
        "log_skewness": round(float(log_series.skew()), 2)
    }


def calculate_price_per_sqft(
    df: pd.DataFrame,
    price_col: str = "price",
    sqft_col: str = "total_sqft"
) -> pd.Series:
    """
    Calculates Price per Square Foot (INR / sq.ft).
    
    WARNING:
    This feature is derived using the target variable 'price'.
    It is strictly designed for EDA and market analytics.
    DO NOT use this as an input feature for ML models to prevent target leakage.
    """
    if price_col not in df.columns or sqft_col not in df.columns:
        raise ValueError(f"Required columns '{price_col}' or '{sqft_col}' missing.")

    # 1 Lakh = 100,000 INR
    return (df[price_col] * 100000.0) / df[sqft_col]


def compute_numerical_summaries(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes 5-number summaries, mean, std, and skewness for all numerical features.
    """
    num_df = df.select_dtypes(include=[np.number])
    summaries = {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        summaries[col] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "q25": round(float(s.quantile(0.25)), 2),
            "q75": round(float(s.quantile(0.75)), 2),
            "max": round(float(s.max()), 2),
            "skewness": round(float(s.skew()), 2)
        }
    return summaries


def compute_categorical_summaries(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Computes frequencies and price distributions per category.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    results = {}
    for col in cat_cols:
        counts = df[col].value_counts().to_dict()
        top_by_price = (
            df.groupby(col)['price'].agg(['count', 'mean', 'median', 'std'])
            .sort_values(by='median', ascending=False)
            .head(10)
            .round(2)
            .to_dict(orient='index')
        ) if 'price' in df.columns else {}

        results[col] = {
            "unique_count": df[col].nunique(),
            "top_categories_by_frequency": {k: counts[k] for k in list(counts.keys())[:10]},
            "top_categories_by_price": top_by_price
        }
    return results


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Computes correlation matrix across numerical variables.
    """
    num_df = df.select_dtypes(include=[np.number])
    return num_df.corr(method=method).round(3)


def compute_location_statistics(
    df: pd.DataFrame,
    location_col: str = "location",
    min_listings: int = 15
) -> Dict[str, Any]:
    """
    Analyzes price and price-per-sqft statistics across metropolitan localities.
    """
    if location_col not in df.columns or "price" not in df.columns:
        return {}

    temp_df = df.copy()
    if "price_per_sqft" not in temp_df.columns and "total_sqft" in temp_df.columns:
        temp_df["price_per_sqft"] = calculate_price_per_sqft(temp_df)

    loc_stats = temp_df.groupby(location_col).agg(
        listing_count=('price', 'count'),
        median_price=('price', 'median'),
        mean_price=('price', 'mean'),
        median_price_per_sqft=('price_per_sqft', 'median'),
        mean_price_per_sqft=('price_per_sqft', 'mean')
    ).reset_index()

    qualified_locs = loc_stats[loc_stats['listing_count'] >= min_listings].sort_values(
        by='median_price_per_sqft', ascending=False
    )

    top_expensive = qualified_locs.head(15).round(2).to_dict(orient='records')
    top_affordable = qualified_locs.tail(15).iloc[::-1].round(2).to_dict(orient='records')

    return {
        "total_unique_locations": int(df[location_col].nunique()),
        "qualified_locations_count": int(len(qualified_locs)),
        "top_15_expensive_localities": top_expensive,
        "top_15_affordable_localities": top_affordable
    }


def compute_outlier_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates IQR outlier bounds across key numerical features.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}

    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = series[(series < low) | (series > high)]

        outlier_report[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_fence": round(float(low), 2),
            "upper_fence": round(float(high), 2),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round((len(outliers) / len(series)) * 100, 2)
        }

    return outlier_report


def generate_eda_visualizations(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> List[str]:
    """
    Generates and saves publication-quality statistical charts.
    """
    out_path = Path(output_dir) if output_dir else VISUALIZATIONS_DIR / "eda"
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    temp_df = df.copy()
    if "price_per_sqft" not in temp_df.columns and "total_sqft" in temp_df.columns:
        temp_df["price_per_sqft"] = calculate_price_per_sqft(temp_df)

    # 1. Target Distribution (Raw vs Log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(temp_df["price"], bins=40, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Target Distribution: Property Price (Raw ₹ Lakhs)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Price (₹ Lakhs)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(temp_df["price"].median(), color="red", linestyle="--", label=f"Median: ₹{temp_df['price'].median():.1f}L")
    axes[0].legend()

    sns.histplot(np.log1p(temp_df["price"]), bins=40, kde=True, ax=axes[1], color="#1b9e77")
    axes[1].set_title("Log-Transformed Target: log(1 + Price)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log(1 + Price in Lakhs)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(["KDE", "Log Histogram"])
    plt.tight_layout()
    f1 = out_path / "target_distribution.png"
    plt.savefig(f1)
    plt.close()
    generated_files.append(str(f1))

    # 2. Total Sqft vs Price Scatter with Trendline
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=temp_df[temp_df["total_sqft"] <= 6000],
        x="total_sqft", y="price", hue="bhk", palette="viridis", alpha=0.6, s=40
    )
    plt.title("Total Built-up Area (Sq. Ft) vs Price (₹ Lakhs)", fontsize=13, fontweight="bold")
    plt.xlabel("Total Area (Sq. Ft)")
    plt.ylabel("Price (₹ Lakhs)")
    plt.xlim(200, 6000)
    plt.ylim(0, 800)
    f2 = out_path / "area_vs_price.png"
    plt.savefig(f2)
    plt.close()
    generated_files.append(str(f2))

    # 3. BHK vs Price Boxplot
    plt.figure(figsize=(10, 6))
    bhk_filtered = temp_df[temp_df["bhk"].between(1, 6)]
    sns.boxplot(data=bhk_filtered, x="bhk", y="price", hue="bhk", palette="Blues_r", showmeans=True, legend=False)
    plt.title("Price Distribution by Bedroom Count (1 to 6 BHK)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Bedrooms (BHK)")
    plt.ylabel("Price (₹ Lakhs)")
    plt.ylim(0, 500)
    f3 = out_path / "price_by_bhk.png"
    plt.savefig(f3)
    plt.close()
    generated_files.append(str(f3))

    # 4. Bathrooms vs Price
    plt.figure(figsize=(10, 6))
    bath_filtered = temp_df[temp_df["bath"].between(1, 6)]
    sns.boxplot(data=bath_filtered, x="bath", y="price", hue="bath", palette="crest", showmeans=True, legend=False)
    plt.title("Price Distribution by Bathroom Count", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Price (₹ Lakhs)")
    plt.ylim(0, 500)
    f4 = out_path / "price_by_bath.png"
    plt.savefig(f4)
    plt.close()
    generated_files.append(str(f4))

    # 5. Area Type vs Price
    if "area_type" in temp_df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=temp_df, x="area_type", y="price", hue="area_type", palette="Set2", showmeans=True, legend=False)
        plt.title("Price Distribution across Area Types", fontsize=13, fontweight="bold")
        plt.xlabel("Area Structure Type")
        plt.ylabel("Price (₹ Lakhs)")
        plt.ylim(0, 400)
        f5 = out_path / "area_type_price_distribution.png"
        plt.savefig(f5)
        plt.close()
        generated_files.append(str(f5))

    # 6. Availability Status vs Price
    if "availability_status" in temp_df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=temp_df, x="availability_status", y="price", hue="availability_status", palette="pastel", showmeans=True, legend=False)
        plt.title("Price Comparison: Ready To Move vs Under Construction", fontsize=13, fontweight="bold")
        plt.xlabel("Construction Delivery Status")
        plt.ylabel("Price (₹ Lakhs)")
        plt.ylim(0, 350)
        f6 = out_path / "availability_status_price.png"
        plt.savefig(f6)
        plt.close()
        generated_files.append(str(f6))

    # 7. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_cols = ["total_sqft", "bath", "balcony", "bhk", "price"]
    available_num = [c for c in num_cols if c in temp_df.columns]
    corr = temp_df[available_num].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Matrix of Numerical Real Estate Attributes", fontsize=12, fontweight="bold")
    f7 = out_path / "correlation_matrix.png"
    plt.savefig(f7)
    plt.close()
    generated_files.append(str(f7))

    # 8. Price Per Sq. Ft Distribution (Capped for visualization)
    plt.figure(figsize=(10, 5))
    psqft_clean = temp_df["price_per_sqft"][temp_df["price_per_sqft"].between(1000, 30000)]
    sns.histplot(psqft_clean, bins=50, kde=True, color="#88419d")
    plt.title("Distribution of Price per Square Foot (₹ / sq.ft)", fontsize=13, fontweight="bold")
    plt.xlabel("Price per Square Foot (₹/sq.ft)")
    plt.ylabel("Property Count")
    plt.axvline(psqft_clean.median(), color="red", linestyle="--", label=f"Median: ₹{psqft_clean.median():.0f}/sq.ft")
    plt.legend()
    f8 = out_path / "price_per_sqft_distribution.png"
    plt.savefig(f8)
    plt.close()
    generated_files.append(str(f8))

    # 9. Top 15 Expensive Localities (Median Price per Sqft)
    loc_summary = temp_df.groupby("location").filter(lambda x: len(x) >= 15)
    if not loc_summary.empty:
        loc_ranks = (
            loc_summary.groupby("location")["price_per_sqft"].median()
            .sort_values(ascending=False).head(15).reset_index()
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(data=loc_ranks, x="price_per_sqft", y="location", hue="location", palette="rocket", legend=False)
        plt.title("Top 15 Prime / Expensive Localities (Median ₹ / Sq. Ft)", fontsize=13, fontweight="bold")
        plt.xlabel("Median Price per Sq. Ft (₹/sq.ft)")
        plt.ylabel("Locality")
        f9 = out_path / "top_expensive_localities_sqft.png"
        plt.savefig(f9)
        plt.close()
        generated_files.append(str(f9))

        # 10. Locality Price Comparison (Median Total Price)
        top_price_locs = (
            loc_summary.groupby("location")["price"].median()
            .sort_values(ascending=False).head(15).reset_index()
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_price_locs, x="price", y="location", hue="location", palette="mako", legend=False)
        plt.title("Top 15 Localities by Median Property Price (₹ Lakhs)", fontsize=13, fontweight="bold")
        plt.xlabel("Median Property Price (₹ Lakhs)")
        plt.ylabel("Locality")
        f10 = out_path / "locality_price_comparison.png"
        plt.savefig(f10)
        plt.close()
        generated_files.append(str(f10))


    logger.info(f"Generated {len(generated_files)} EDA plots in {out_path}")
    return generated_files


def run_full_eda(
    data_path: Path = CLEANED_DATA_PATH,
    output_vis_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end exploratory analysis and generates all statistical artifacts.
    """
    logger.info(f"Running Full EDA on dataset: {data_path}")
    df = pd.read_csv(data_path)

    overview = compute_dataset_overview(df)
    target_stats = compute_target_statistics(df, target_col="price")
    num_stats = compute_numerical_summaries(df)
    cat_stats = compute_categorical_summaries(df)
    corr_matrix = compute_correlation_matrix(df).to_dict()
    location_stats = compute_location_statistics(df, min_listings=15)
    outlier_stats = compute_outlier_summary(df)

    vis_files = generate_eda_visualizations(df, output_dir=output_vis_dir)

    return {
        "overview": overview,
        "target_statistics": target_stats,
        "numerical_summaries": num_stats,
        "categorical_summaries": cat_stats,
        "correlation_matrix": corr_matrix,
        "location_insights": location_stats,
        "outlier_analysis": outlier_stats,
        "generated_visualizations": vis_files
    }


if __name__ == "__main__":
    results = run_full_eda()
    print("\n" + "="*60)
    print("SMARTHOUSE AI - EDA SUMMARY REPORT")
    print("="*60)
    print(f"Total Properties Analyzed: {results['overview']['total_rows']}")
    print(f"Target Price: Mean = INR {results['target_statistics']['mean']}L, Median = INR {results['target_statistics']['median']}L")
    print(f"Target Skewness: {results['target_statistics']['skewness']} (Log Skewness: {results['target_statistics']['log_skewness']})")
    print(f"Localities Identified: {results['location_insights']['total_unique_locations']}")
    print(f"Plots Generated: {len(results['generated_visualizations'])}")
    print("="*60)

