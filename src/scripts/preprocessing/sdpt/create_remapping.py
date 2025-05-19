import os
import pandas as pd
import logging
import src.scripts.preprocessing.preprocessing_constants as cn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define classification dictionaries
short_rotation_species = ['Acacia', 'Eucalyptus', 'Poplar', 'Willow', 'Wood fiber', 'Timber']
long_rotation_species = ['Rubber', 'Fruit', 'Teak', 'Pine', 'Spruce', 'Fir']
oil_palm_species = ['Oil palm']

# Helper function for rough matching
def rough_classification(name):
    name = name.lower()
    if any(spec.lower() in name for spec in short_rotation_species):
        return 'short_rotation'
    elif any(spec.lower() in name for spec in long_rotation_species):
        return 'long_rotation'
    elif 'oil palm' in name:
        return 'oil_palm'
    else:
        return 'long_rotation'  # Default conservative approach

# Main processing function
def remap_sdpt_attributes(input_csv, output_csv):
    logging.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)

    # Initial classification based on provided attributes
    logging.info("Applying initial classification based on simpleType and vernacName")
    def classify(row):
        if row['simpleType'] == 'Tree crops' and row['vernacName'] in oil_palm_species:
            return 'oil_palm'
        elif row['simpleType'] == 'Planted forest' or (row['simpleType'] == 'Tree crops' and row['vernacName'] not in oil_palm_species):
            return rough_classification(row['vernacName'])
        else:
            return 'long_rotation'

    df['rotation_class'] = df.apply(classify, axis=1)

    logging.info("Reclassification complete. Reviewing results:")
    logging.info(df['rotation_class'].value_counts())

    logging.info(f"Saving reclassified data to {output_csv}")
    df.to_csv(output_csv, index=False)

# Example usage
if __name__ == "__main__":
    input_csv = os.path.join(
        cn.local_root,
        "Plantation",
        "plantation_attributes_v21_v09152024_FINAL.csv",
    )
    output_csv = os.path.join(cn.local_root, "Plantation", "rough_remapping.csv")

    remap_sdpt_attributes(input_csv, output_csv)
