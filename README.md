# Twitter Sentiment Analysis

## Project Overview

This project analyses sentiment on Twitter data to understand public opinion and attitudes towards specific topics or brands. The analysis includes data preprocessing, feature extraction using TF-IDF, and sentiment classification using machine learning models, specifically Support Vector Machine (SVM).

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling and Analysis](#modeling-and-analysis)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Data

The dataset used in this project is sourced from Kaggle, specifically the [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) dataset. It consists of two CSV files:

- `twitter_training.csv`: Training data containing tweets labeled with sentiment.
- `twitter_validation.csv`: Validation data used for model testing.

Each dataset contains the following columns:

- `id`: Unique identifier for each tweet.
- `game`: The topic or brand mentioned in the tweet.
- `sentiment`: The sentiment label (Positive, Negative, Neutral, Irrelevant).
- `text`: The content of the tweet.

## Installation

To run this project, ensure you have Python 3.7+ installed along with the required libraries. You can install the necessary dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Load and preprocess the data**:
   - Load the datasets `twitter_training.csv` and `twitter_validation.csv`.
   - Clean the data by removing missing values in the `text` column.
   - Encode categorical variables (sentiment and game) using label encoding.

3. **Feature Extraction**:
   - Convert the tweet text into numerical features using TF-IDF vectorization.
   - Use the top 5000 features from the text data for model training.

4. **Model Training**:
   - Train a Support Vector Machine (SVM) model using the training data.
   - Evaluate the model using the validation data.

5. **Results**:
   - Evaluate the performance of the SVM model using accuracy, classification report, and confusion matrix.
   - Visualize the confusion matrix to understand the model's performance.

6. **Hyperparameter Tuning**:
   - Optionally, use GridSearchCV to find the optimal hyperparameters for the SVM model.

## Modeling and Analysis

The project involves the following steps:

1. **Data Preprocessing**:
   - Handling missing data and encoding categorical variables.
   
2. **TF-IDF Vectorization**:
   - Transforming text data into numerical features for model training.
   
3. **SVM Model Training**:
   - Training an SVM classifier with a linear kernel on the processed data.
   
4. **Evaluation**:
   - Using metrics such as accuracy, precision, recall, and F1-score to evaluate the model.
   - Visualizing the results using confusion matrix plots.

## Results

The SVM model achieved [insert accuracy]% accuracy on the validation set. The model effectively classified the sentiment of tweets into four categories: Positive, Negative, Neutral, and Irrelevant.

### Example Results:
- **Accuracy**: [insert accuracy]%
- **Confusion Matrix**:
  ![Confusion Matrix] (Screenshot 2024-08-13 104925.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
