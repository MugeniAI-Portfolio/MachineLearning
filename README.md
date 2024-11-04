# Sentiment Analysis of Amazon and IMDb Reviews Using Logistic Regression

## Project Overview
This project showcases the process of building and evaluating a Logistic Regression model for sentiment analysis on both Amazon and IMDb review datasets. The goal is to classify reviews as positive, neutral, or negative, leveraging text vectorization and machine learning techniques. The project provides insights into how machine learning models can be effectively used to understand customer sentiment and enhance business strategies.

## Datasets
### Amazon Reviews
- **Source**: Kaggle
- **Features**: Review text, labeled with sentiment.
- **Classes**: Positive, Neutral, Negative.
- **Class Distribution**:
  - **Before balancing**: Significant imbalance with more positive samples.
  - **After balancing**: Balanced distribution across all classes.

### IMDb Reviews
- **Source**: [Large Movie Review Dataset (Maas et al., 2011)](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Features**: 25,000 positive and 25,000 negative reviews for training; additional 25,000 for testing.
- **Classes**: Positive, Negative.
- **Class Characteristics**:
  - Average review length: ~144 words.
  - Provides diverse expressions and sentiments for robust training.

## Data Preprocessing
1. **Cleaning and Preparing Data**:
   - Removed null values and selected relevant columns.
2. **Text Cleaning**:
   - Removed special characters and numbers.
   - Converted text to lowercase.
   - Stripped extra whitespace.
3. **Label Conversion**:
   - Converted sentiment scores to categorical labels for classification.

## Feature Extraction
- **Method**: TF-IDF Vectorization.
- **Purpose**: Transform raw text data into numerical vectors suitable for model training.

## Model Training
- **Algorithm**: Logistic Regression.
- **Hyperparameters**:
  - `max_iter`: Set to 5000 for sufficient convergence.
  - `class_weight`: Balanced to manage class imbalances.
- **Training-Testing Split**: 80% training, 20% testing.

## Model Evaluation
### Metrics Used
- Accuracy.
- Precision, Recall, and F1-score.
- Confusion Matrix for visual analysis of class performance.

### Results
- **Amazon Reviews**:
  - Overall accuracy: 76%.
  - Strong performance for positive and negative classes; the neutral class shows room for improvement.
- **IMDb Reviews**:
  - Validation accuracy: 87.28%.
  - Test accuracy: 84.88%.
  - AUC score: 0.94, indicating strong predictive performance.

### Visualizations
- **Confusion Matrix**: Displayed as a heatmap using Seaborn for interpretability.
- **ROC Curve**: Used to illustrate the trade-off between true positive and false positive rates.

## Final Thoughts
### Challenges
- Handling long reviews while maintaining context.
- Lower recall for neutral classes in Amazon reviews.
- Detecting nuances like sarcasm and mixed sentiments in IMDb reviews.

### Next Steps
- Experiment with more complex models like SVM and Random Forest.
- Implement deep learning architectures (e.g., LSTM, BERT) for enhanced context understanding.
- Apply transfer learning for leveraging pre-trained models on related tasks.

## How to Run This Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```
2. **Install Dependencies**:
   Ensure Python 3 and pip are installed.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**:
   Launch the Jupyter notebook or run the Python script for step-by-step execution.

## References
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
- Google Developers. *Text Classification Guide*. [Link](https://developers.google.com/machine-learning/guides/text-classification)
