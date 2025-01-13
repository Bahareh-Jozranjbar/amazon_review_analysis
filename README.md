# Amazon Review Analysis with LSTM

This project demonstrates how to analyze Amazon reviews using an **LSTM-based neural network**. It covers text preprocessing, integration of pre-trained **GloVe embeddings**, model training, and performance evaluation. The focus is on exploring **deep learning techniques** for sentiment analysis, making it an excellent project for researchers and developers alike.

---

## Project Structure

```plaintext
amazon_review_analysis/
├── data/                   # Contains processed reviews and test data
│   ├── processed_reviews.csv   # Training dataset
│   ├── test_data.csv           # Testing dataset
├── models/                # Stores trained models and tokenizers
│   ├── lstm_sentiment_model.h5   # Trained model in HDF5 format
│   ├── lstm_sentiment_model.keras # Trained model in Keras format
│   ├── tokenizer.pkl           # Tokenizer for text preprocessing
├── scripts/               # Python scripts for training, evaluation, and preprocessing
│   ├── DeepLearningSentiment.py  # Trains the LSTM model
│   ├── EvaluateModel.py         # Evaluates the trained model
│   ├── PrepareTestData.py       # Prepares the test dataset
├── README.md              # Project documentation
├── requirements.txt       # List of Python package dependencies
```

## Getting Started
### Prerequisites
Python 3.8 or later
TensorFlow 2.x
Numpy, Pandas, Scikit-learn, Matplotlib
### Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/amazon_review_analysis.git
cd amazon_review_analysis
```

2. Create and activate a virtual environment:

```bash
# Linux/Mac
python3 -m venv env
source env/bin/activate

# Windows
python -m venv env
.\env\Scripts\activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```


## Running the Project

1. Train the LSTM Model
Run the training script to preprocess data, integrate GloVe embeddings, and train the LSTM model:

```bash
python scripts/DeepLearningSentiment.py

```
2. Prepare Test Data
Prepare the test dataset for evaluation:

```bash
python scripts/PrepareTestData.py
```

3. Evaluate the Model
Evaluate the trained LSTM model on the test dataset and generate a classification report:

```bash
python scripts/EvaluateModel.py
```

### Features

1. Preprocessing:

- Text tokenization and padding.
- Integration of GloVe word embeddings for improved semantic understanding.

2. Deep Learning:

- LSTM-based model with dropout layers for regularization.
- Early stopping and class balancing for efficient training.

3. Evaluation:

- Generates accuracy metrics and classification reports to measure model performance.


### Example Output:

| Metric       | Negative | Neutral | Positive |
|--------------|----------|---------|----------|
| **Precision**| 15%      | 7%      | 78%      |
| **Recall**   | 15%      | 15%     | 70%      |
| **F1-Score** | 15%      | 10%     | 74%      |

### Future Enhancements
- Experiment with transformer models like BERT for more advanced sentiment analysis.
- Implement data augmentation to address class imbalance.
- Fine-tune model hyperparameters for better generalization.

### License
This project is licensed under the MIT License. Feel free to use, share, and modify it as needed.
