import os
import pandas as pd
import logging
import src.scripts.preprocessing.preprocessing_constants as cn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Global classification lists for readability and maintainability
OIL_PALM_INDICATORS = ['oil palm', 'elaeis guineens']
LONG_ROTATION_SIMPLE_NAMES = ['fruit', 'rubber']
SHORT_ROTATION_SCI_NAMES = [
    'acacia', 'eucalyptus', 'alnus', 'populus', 'salix',
    'bambusa oldhamii', 'guadua angustifolia', 'gmelina arborea',
    'gliricidia sepium', 'schizolobium', 'musa', 'casuarina',
    'melia azedarach', 'paulownia', 'prosopis', 'robinia', 'triplaris',
    'ochroma pyramidale', 'leucaena', 'unknown', 'bamboo'
]

# Class code dictionary
ROTATION_CLASS_CODES = {
    'oil_palm': 1,
    'short_rotation': 2,
    'long_rotation': 3
}

def classify(row):
    """Hierarchical classification logic based on plantation type."""
    simple_name = str(row['simpleName']).lower()
    sci_name = str(row['sciName']).lower()

    # Oil Palm classification (highest priority)
    if any(keyword in simple_name for keyword in OIL_PALM_INDICATORS):
        return 'oil_palm'

    # Long rotation classification from simple names
    if any(keyword in simple_name for keyword in LONG_ROTATION_SIMPLE_NAMES):
        return 'long_rotation'

    # Oil Palm detailed classification from scientific names
    if any(keyword in sci_name for keyword in OIL_PALM_INDICATORS):
        return 'oil_palm'

    # Short rotation detailed classification from scientific names
    if any(keyword in sci_name for keyword in SHORT_ROTATION_SCI_NAMES):
        return 'short_rotation'

    # Default conservative assignment
    return 'long_rotation'

def remap_sdpt_attributes(input_csv, output_csv):
    """Load data, apply classification, and save results with numeric codes."""
    logging.info(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    logging.info("Applying plantation classification logic.")
    df['rotation_class'] = df.apply(classify, axis=1)

    # Map class names to numeric codes
    df['rotation_code'] = df['rotation_class'].map(ROTATION_CLASS_CODES)

    logging.info("Classification completed. Summary of classifications:")
    logging.info(f"\n{df['rotation_class'].value_counts()}")

    logging.info("Numeric classification code summary:")
    logging.info(f"\n{df['rotation_code'].value_counts()}")

    logging.info(f"Saving reclassified data to: {output_csv}")
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = os.path.join(
        cn.local_root,
        "Plantation",
        "plantation_attributes_v21_v09152024_FINAL.csv"
    )
    output_csv = os.path.join(
        cn.local_root,
        "Plantation",
        "advanced_remapping.csv"
    )

    remap_sdpt_attributes(input_csv, output_csv)
