import pandas as pd
import sys
from pathlib import Path

# Import validation functions
from validation import(
    inspect_misclassifications,
    validation_confused_pairs,
    export_validation_sample,
    quick_lookup
)

# Import data processing to get confusion examples
from data_processing import (
    load_raw_data,
    feature_engineering,
    initial_ml_clustering,
    run_business_analysis
)

def main():
    """Run validation workflow"""
    print("Loading data and running analysis...")

    # Load and process
    df = load_raw_data()
    df_ml, tfidf_features, phrase_features, tfidf_vec, phrase_vec = feature_engineering(df)

    # Run models
    model_tfidf, model_phrases, model_combined, meta_model, df_filtered, X_tfidf, X_phrases, y = initial_ml_clustering(
        df_ml, tfidf_features, phrase_features
    )

    # Get business analysis with confusion  tracking
    # Will need to modify run_business_analysis to return confusion_examples
    # For initial testing purposes, will create here:

    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model_tfidf, X_tfidf, y, cv=5)

    # Build confusion examples dictionary
    confusion_examples = {}
    for i, (true_cat, pred_cat) in enumerate(zip(y, y_pred)):
        if true_cat != pred_cat:
            pair = (true_cat, pred_cat)
            if pair not in confusion_examples:
                confusion_examples[pair] = []
            confusion_examples[pair].append(df_filtered.index[i])

    print("\n"+"="*80)
    print("VALIDATION MENU")
    print("="*80)
    print("1. Inspect specific confused pair")
    print("2. Review all top confused paris")
    print("3. Export validation samples to CSV")
    print("4. Quick lookup by index")
    print("5. Exit")

    while True:
        choice = input("\nEnter choice (1-5): ")

        if choice == "1":
            print("\nAvailable confused pairs:")
            for i, (pair, indices) in enumerate(list(confusion_examples.items())[:10]):
                print(f"{i+1}. {pair[0]} -> {pair[1]} ({len(indices)} cases)")

            pair_choice = int(input("Select pair number: ")) - 1
            selected_pair = list(confusion_examples.items())[pair_choice]
            true_cat, pred_cat = selected_pair[0]
            indices = selected_pair[1]

            inspect_misclassifications(df, indices, true_cat, pred_cat)

        elif choice == "2":
            validation_confused_pairs(df, confusion_examples)

        elif choice == "3":
            output_file = input("Output filename (default: validation_samples.csv): ").strip()
            if not output_file:
                output_file = "validation_samples.csv"
            export_validation_sample(df, confusion_examples, output_file)

        elif choice == "4":
            idx = int(input("Enter index number: "))
            quick_lookup(df, idx)

        elif choice == "5":
            print("Exiting validation tool.")
            break

        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
