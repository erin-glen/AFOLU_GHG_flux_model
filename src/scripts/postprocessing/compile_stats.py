"""
Compare organic‑soil emissions from your model with FAOSTAT.

Requires:
    • iso_lookup.py  (same directory) – provides ISO_LOOKUP dict
    • pandas ≥ 2.1
    • openpyxl       (for reading the Excel model output)

Outputs tidy & wide CSVs in ./comparison_outputs
"""
from pathlib import Path
import pandas as pd
from iso_lookup import ISO_LOOKUP   # <-- externalised mapping

# ---------------------------------------------------------------------
MODEL_EXCEL = r"C:\tmp\0724_stats\drainage_model_1x1_chunk_statistics_20250724_20_14_28.xlsx"
FAO_CSV     = r"C:\tmp\0724_stats\FAOSTAT_data_en_7-25-2025.csv"
# ---------------------------------------------------------------------

PERIODS = {
    "2001_2005": (2001, 2005),
    "2006_2010": (2006, 2010),
    "2011_2015": (2011, 2015),
    "2016_2020": (2016, 2020),
    "2021_2024": (2021, 2024),
}
PERIOD_LEN = {p: end - start + 1 for p, (start, end) in PERIODS.items()}


def year_to_period(year: int) -> str | None:
    for label, (start, end) in PERIODS.items():
        if start <= year <= end:
            return label
    return None


# -------------------- model loader -----------------------------------
def load_model_outputs(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        sheet_name="other_outputs_1x1",
        usecols=["iso", "layer_name", "years", "sum_value"],
    )

    keep_layers = {
        "drained_total_Mg_CO2e_ha_yr": "drained",
        "burned_total_Mg_CO2e_ha": "burned",
    }
    df = df[df["layer_name"].isin(keep_layers)]
    df["emission_type"] = df["layer_name"].map(keep_layers)

    df["period_len"] = df["years"].map(PERIOD_LEN)
    burned = df["emission_type"] == "burned"
    df.loc[burned, "sum_value"] = (
        df.loc[burned, "sum_value"] / df.loc[burned, "period_len"]
    )

    df = (
        df.rename(columns={"years": "period", "sum_value": "annual_Mg_CO2e"})
        .assign(source="model")
        .loc[:, ["iso", "period", "emission_type", "annual_Mg_CO2e", "source"]]
        .sort_values(["iso", "period", "emission_type"])
    )
    return df


# -------------------- FAOSTAT loader ---------------------------------
def load_faostat(path: str | Path) -> pd.DataFrame:
    item_map = {
        "Drained organic soils": "drained",
        "Fires in organic soils": "burned",
    }

    df = pd.read_csv(path, usecols=["Area", "Item", "Year", "Value"])

    df["iso"] = df["Area"].map(ISO_LOOKUP)
    df = df[df["iso"].notna()]  # drop aggregates

    df["emission_type"] = df["Item"].map(item_map)
    df = df[df["emission_type"].notna()]

    df["annual_Mg_CO2e"] = df["Value"] * 1_000
    df["period"] = df["Year"].apply(year_to_period)
    df = df.dropna(subset=["period"])

    df = (
        df.groupby(["iso", "period", "emission_type"], as_index=False)
        ["annual_Mg_CO2e"]
        .mean()
        .assign(source="FAOSTAT")
    )
    return df


# -------------------- main driver ------------------------------------
def main() -> None:
    model_df = load_model_outputs(MODEL_EXCEL)
    fao_df   = load_faostat(FAO_CSV)

    combined_long = pd.concat([model_df, fao_df], ignore_index=True)
    combined_wide = (
        combined_long.pivot_table(
            index=["iso", "period", "emission_type"],
            columns="source",
            values="annual_Mg_CO2e",
        )
        .reset_index()
    )

    out_dir = Path("comparison_outputs")
    out_dir.mkdir(exist_ok=True)
    combined_long.to_csv(out_dir / "model_vs_faostat_long.csv", index=False)
    combined_wide.to_csv(out_dir / "model_vs_faostat_wide.csv", index=False)
    print("✓ comparison tables written to", out_dir.resolve())


if __name__ == "__main__":
    main()
