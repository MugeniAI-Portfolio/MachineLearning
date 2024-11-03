# Sentiment Analysis of Amazon Reviews Using Logistic Regression

## Project Overview
This project demonstrates the process of building and evaluating a Logistic Regression model for sentiment analysis on Amazon review data. The main objective is to classify reviews as positive, neutral, or negative using text vectorization and machine learning techniques.

## Dataset
- **Source**: Kaggle.
- **Features**: Review text, labeled with sentiment.
- **Classes**: Positive, Neutral, Negative.
- **Class Distribution**:
  - Before balancing: significant imbalance with more positive samples.
  - After balancing: equal number of samples for all classes.

## Data Preprocessing
1. **Cleaning and Preparing Data**:
   - Removed null values.
   - Selected relevant columns (`Text` and `Sentiment`).
2. **Text Cleaning**:
   - Removed special characters and numbers.
   - Converted text to lowercase.
   - Stripped extra whitespace.
3. **Label Conversion**:
   - Sentiment scores were converted to categorical labels (positive, neutral, negative).

## Feature Extraction
- **Method**: TF-IDF Vectorization.
- **Purpose**: Transform textual data into numerical vectors for model training.

## Model Training
- **Algorithm**: Logistic Regression.
- **Hyperparameters**:
  - `max_iter`: Set to 5000 for sufficient convergence.
  - `class_weight`: Balanced to handle class imbalance.
- **Training-Testing Split**: 80% training, 20% testing.

## Model Evaluation
- **Metrics Used**:
  - Accuracy.
  - Precision, Recall, and F1-score.
  - Confusion Matrix for a visual representation of class-wise performance.
- **Results**:
  - Overall accuracy: 76%.
  - Good performance for positive and negative classes; neutral class shows scope for improvement.

## Visualization
- **Confusion Matrix**: Displayed as a heatmap using Seaborn for better interpretability.

## Final Thoughts
- **Challenges**: The model performs well on positive and negative classes but has lower recall for neutral reviews.
- **Next Steps**:
  - Consider using more complex models (e.g., SVM, Random Forest).
  - Experiment with deep learning models like LSTM or BERT for potentially better performance.

## How to Run This Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/amazon-review-sentiment-analysis.git
   cd amazon-review-sentiment-analysis
   ```
2. **Install Dependencies**:
   Ensure you have Python 3 and pip installed.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**:
   Open the Jupyter notebook or run the Python script.

## References
- **Scikit-learn Documentation**: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- **TF-IDF**: [Understanding TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- **Dataset Source**: [Amazon Review Dataset](https://link-to-dataset.com](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews?resource=download)
