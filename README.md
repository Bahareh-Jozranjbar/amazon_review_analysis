# Amazon Review Analysis with LSTM

This project demonstrates how to analyze Amazon reviews using an **LSTM-based neural network**. It covers text preprocessing, integration of pre-trained **GloVe embeddings**, model training, and performance evaluation. The focus is on exploring **deep learning techniques** for sentiment analysis, making it an excellent project for researchers and developers alike.

---

## Project Structure

```plaintext
amazon_review_analysis/
├── data/
│   ├── Reviews.csv                # Raw dataset (not uploaded to GitHub)
│   ├── processed_reviews.csv      # Preprocessed dataset (generated locally)
├── scripts/
│   ├── PreprocessReviews.py       # Script to preprocess the raw dataset
│   ├── PrepareTestData.py         # Script to prepare test data
│   ├── DeepLearningSentiment.py   # Script for training the LSTM model
│   ├── EvaluateModel.py           # Script to evaluate the trained model
├── models/
│   ├── tokenizer.pkl              # Saved tokenizer
│   ├── lstm_sentiment_model.h5    # Trained model file
│   ├── lstm_sentiment_model.keras # Trained model file in Keras format
├── README.md                      # Documentation

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

###  Dataset Preparation
1. Download the Dataset
Download the Amazon Fine Food Reviews Dataset. You can download it from:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

2. Save the Dataset
Place the downloaded file Reviews.csv in the data/ directory of the project.

3. Generate Preprocessed Data
Run the preprocessing script to create the preprocessed dataset:

```bash
python scripts/PreprocessReviews.py
```

4. Result
The script will generate a file data/processed_reviews.csv, which is used as input for the model training and evaluation.

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
