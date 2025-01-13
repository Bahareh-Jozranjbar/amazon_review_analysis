import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Step 1: Load Preprocessed Data
data_file = "data/processed_reviews.csv"  # Path to your processed dataset
df = pd.read_csv(data_file)

# Split data into text and labels
texts = df["Text"].values
labels = df["Sentiment"].map({"Positive": 2, "Neutral": 1, "Negative": 0}).values

# Step 2: Tokenize and Pad Sequences
max_vocab = 5000
max_length = 100

tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Step 3: Load GloVe Embeddings
embedding_dim = 100
glove_path = "glove.6B/glove.6B.100d.txt"  # Path to your GloVe file
embedding_index = {}

# IMPORTANT: Ensure you download the GloVe embeddings from the official source:
# https://nlp.stanford.edu/projects/glove/
# Unzip the downloaded file and place the `glove.6B.100d.txt` file in the `glove.6B` folder.

print("Loading GloVe embeddings...")
with open(glove_path, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((max_vocab, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_vocab:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Step 4: Calculate Class Weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

# Step 5: Define the Model
model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=embedding_dim, weights=[embedding_matrix],
              input_length=max_length, trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 6: Train the Model with Class Weights and Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Step 7: Save the Model
model_path = "models/lstm_sentiment_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

# Step 8: Save the Training History
history_file = "models/history.json"
with open(history_file, "w") as f:
    json.dump(history.history, f)
print(f"Training history saved to {history_file}")

# Step 9: Plot Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Step 10: Save the Model in Keras Format
model_keras_path = "models/my_model.keras"
model.save(model_keras_path)
print(f"Model saved in Keras format to: {model_keras_path}")
