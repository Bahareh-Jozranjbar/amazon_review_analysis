import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Step 1: Load the Processed Reviews Dataset
data_file = "data/processed_reviews.csv"  # Path to the processed reviews dataset
print("Loading the processed dataset...")
df = pd.read_csv(data_file)

# Split the data into features (Text) and labels (Sentiment)
texts = df["Text"].values
labels = df["Sentiment"].map({"Positive": 2, "Neutral": 1, "Negative": 0}).values

# Step 2: Tokenize and Pad the Texts
print("Tokenizing and padding texts...")
max_vocab = 5000  # Maximum number of words to keep in the vocabulary
max_length = 100  # Maximum length of sequences

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save the tokenizer
tokenizer_path = "models/tokenizer.pkl"
print(f"Saving tokenizer to {tokenizer_path}...")
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

# Convert texts to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# Step 3: Split into Training and Test Sets
print("Splitting the data into training and test sets...")
_, test_texts, _, test_labels = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Step 4: Save the Test Data
print("Saving the test data...")
test_data_df = pd.DataFrame({
    "Text": [text for text in texts[len(texts) - len(test_texts):]],
    "Label": test_labels
})
test_data_path = "data/test_data.csv"
test_data_df.to_csv(test_data_path, index=False)
print(f"Test data recreated and saved to {test_data_path}!")
