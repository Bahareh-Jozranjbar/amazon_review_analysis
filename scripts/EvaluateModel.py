import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Step 1: Load the Test Data
test_data_path = "data/test_data.csv"  # Path to the prepared test dataset
print("Loading test data...")
test_data_df = pd.read_csv(test_data_path)

# Extract features (Text) and labels (Label)
test_texts = test_data_df["Text"].values
test_labels = test_data_df["Label"].values  # Labels are integer-encoded (0: Negative, 1: Neutral, 2: Positive)

# Step 2: Load the Tokenizer
tokenizer_path = "models/tokenizer.pkl"  # Path to the saved tokenizer
print("Loading tokenizer...")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Step 3: Tokenize and Pad the Test Data
print("Tokenizing and padding test data...")
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=100)  # Ensure max length matches training

# Step 4: Load the Trained Model
model_path = "models/lstm_sentiment_model.keras"  # Path to the saved model
print("Loading the trained model...")
model = load_model(model_path)

# Step 5: Evaluate the Model
print("Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 6: Predict and Generate a Classification Report
print("\nGenerating predictions and classification report...")
predicted_probs = model.predict(test_data)
predicted_labels = np.argmax(predicted_probs, axis=1)

# Generate classification report
target_names = ["Negative", "Neutral", "Positive"]
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, target_names=target_names))
