import pandas as pd
import numpy as np
from pathlib import Path

def inspect_misclassifications(df, indices, true_category, pred_category, max_display=5):
    """
    Display detailed information about misclassified incidents.

    Parameters:
    -----------
    :param df: DataFrame
        Original dataframe with incident reports
    :param indices: list
        List of row indices to inspect
    :param true_category: str
        The actual failure category
    :param pred_category: str
        What the model predicted
    :param max_display: int
        Max number of examples to display
    """

    print("="*80)
    print(f"MISCLASSIFICATION ANALYSIS")
    print(f"True Category: {true_category}")
    print(f"Predicted as: {pred_category}")
    print(f"Total misclassifications: {len(indices)}")
    print("="*80)

    for i, idx in enumerate(indices[:max_display]):
        print(f"\n[Example {i+1}] - Index: {idx}")
        print("-"*80)

        # Get record
        record = df.loc[idx]

        print(f"Report Text:")
        print(f"    {record['Report'][:300]}...")
        print(f"\nActual Classification: {record['Part failure']}")
        print(f"Model Prediction: {pred_category}")
        print(f"\nOther Fields:")
        print(f"    Nature/Condition: {record['Occurence Nature Condition']}")
        print(f"    Precautionary Procedures: {record['Occurence Precautionary Procedures']}")

    if len(indices) > max_display:
        print(f"\n... and {len(indices) - max_display} more exmaples")
        print(f"Full index list: {indices}")

def validation_confused_pairs(df, confusion_examples, top_n=5):
    """
    Validate the most confused classification pairs

    Parameters:
    -----------
    :param df: DataFrame
        Original dataframe
    :param confusion_examples: dict
        Dictionary mapping (true_cat, pred_cat) tuples to lists of indices
    :param top_n: int
        Number of confused pairs to display
    """

    print("\n"+"="*80)
    print("TOP CONFUSED PAIRS - DETAILED INSPECTION")
    print("="*80)

    # Sort by number of confusions
    sorted_pairs = sorted(confusion_examples.items(),
                          key=lambda x: len(x[1]),
                          reverse=True)

    for (true_cat, pred_cat), indices in sorted_pairs[:top_n]:
        print(f"\n{'='*80}")
        inspect_misclassifications(df, indices, true_cat, pred_cat, max_display=3)

def export_validation_sample(df, confusion_examples, output_path="validation_samples.csv"):
    """
    Export a CSV of misclassified examples for external review.

    Parameters:
    -----------
    :param df: DataFrame
        Original dataframe
    :param confusion_examples: dict
        Dictionary mapping (true_cat, pred_cat) tuples to lists of indices
    :param output_path: str
        Path to save the CSV file
    """

    validation_records = []

    for (true_cat, pred_cat), indices in confusion_examples.items():
        for idx in indices[:10]:    # Limit to 10 per pair
            record = df.loc[idx].copy()
            validation_records.append({
                'Index': idx,
                'Report': record['Report'],
                'True_Category': true_cat,
                'Predicted_Category': pred_cat,
                'Nature_Condition': record['Occurence Nature condition'],
                'Precautionary_Procedures': record['Occurence Precautionary Procedures']
            })

    validation_df = pd.DataFrame(validation_records)
    validation_df.to_csv(output_path, index=False)
    print(f"\nExported {len(validation_df)} validation samples to {output_path}")
    return validation_df

def quick_lookup(df, index):
    """
    Quick lookup of a single incident by index

    Parameters:
    -----------
    :param df: DataFrame
        Original dataframe
    :param index: int
        Row index to look up
    """

    print(f"\n{'='*80}")
    print(f"INCIDENT LOOKUP - Index: {index}")
    print("="*80)

    record = df.loc[index]

    print(f"\nPart failure Classification: {record['Part Failure']}")
    print(f"\nFull Report:")
    print(record['Report'])
    print(f"\nNature/Conditions: {record['Occurence Nature condition']}")
    print(f"Precautionary Procedures: {record['Occurence Precautionary Procedures']}")

if __name__ == "__main__":
    # Example usage
    print("Validation module loaded.")
    print("Import this module and use the following functions:")
    print("  - inspect_misclassifications(df, indices, true_cat, pred_cat)")
    print("  - validate_confused_pairs(df, confusion_examples)")
    print("  - export_validation_sample(df, confusion_examples)")
    print("  - quick_lookup(df, index)")
