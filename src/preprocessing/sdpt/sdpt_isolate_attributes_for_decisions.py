import os
import pandas as pd

def main():
    # 1) Path to your input CSV
    csv_path = r"C:\Users\Erin.Glen\Downloads\plantation_attributes_v2_draft_v11282023.csv"

    # 2) Path to your output CSV for unique species
    out_csv = r"C:\Users\Erin.Glen\Downloads\unique_planted_forest_species.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    # 3) Load data
    df = pd.read_csv(csv_path)

    # 4) Filter rows where simpleType == "Planted forest"
    df_planted = df[df["simpleType"] == "Planted forest"]

    # 5) Collect unique vernacName values (dropping any NaN)
    unique_species = df_planted["vernacName"].dropna().unique()
    unique_species = sorted(unique_species)

    # 6) Print results
    print("Unique vernacName for rows with simpleType == 'Planted forest':")
    if unique_species:
        for name in unique_species:
            print(f"  - {name}")
    else:
        print("  (No matching rows found!)")

    # 7) Export to CSV
    if unique_species:
        # Convert list to a small DataFrame with one column
        df_unique = pd.DataFrame(unique_species, columns=["vernacName"])
        df_unique.to_csv(out_csv, index=False)
        print(f"\nUnique vernacName values exported to: {out_csv}")
    else:
        print("\nNo data to export.")

if __name__ == "__main__":
    main()
