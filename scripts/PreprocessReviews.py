"""
Preprocess Reviews Dataset

This script preprocesses the Amazon Fine Food Reviews dataset by:
1. Selecting relevant columns (`Score` and `Text`).
2. Generating sentiment labels (`Positive`, `Neutral`, `Negative`) based on the `Score`.
3. Saving the preprocessed data to `data/processed_reviews.csv`.

Note: Ensure the raw dataset `Reviews.csv` is downloaded and placed in the `data/` directory.
 You can download it from:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
"""

import pandas as pd

# Path to the raw dataset (update if necessary)
file_path = "data/Reviews.csv"

# Load the raw dataset
print("Loading the dataset...")
df = pd.read_csv(file_path)

# Select relevant columns and drop missing values
print("Preprocessing the dataset...")
df = df[["Score", "Text"]].dropna()

# Function to create sentiment labels
def assign_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

# Apply the sentiment function
df["Sentiment"] = df["Score"].apply(assign_sentiment)

# Save the preprocessed data
output_file_path = "data/processed_reviews.csv"
df.to_csv(output_file_path, index=False)

print(f"Preprocessed data saved to: {output_file_path}")
