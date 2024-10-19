
# Fake News Detection with Machine Learning

This project is designed to detect fake news using various machine learning models. It uses text-based news data to predict whether a news article is **fake** or **factual** based on its content. The models implemented include **Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier**, and **Gradient Boosting Classifier**.

## Dataset

The dataset used in this project consists of two CSV files:
1. **True.csv**: Contains factual news articles.
2. **Fake.csv**: Contains fake news articles.

Both datasets have the following columns:
- **title**: The title of the news article (not used in this project).
- **text**: The content of the news article.
- **subject**: The category of the news article (not used in this project).
- **date**: The publication date of the news article (not used in this project).

The project labels the data:
- `1` for **true news**.
- `0` for **fake news**.

You can download these datasets from public fake news datasets like the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

## Project Overview

This project involves the following steps:
1. **Data Loading**: The `True.csv` and `Fake.csv` datasets are read, labeled, and combined into a single dataset.
2. **Data Preprocessing**: The text of each article is cleaned and processed to remove unwanted characters such as URLs, HTML tags, punctuation, and digits.
3. **Feature Engineering**: The text data is transformed into numerical features using **TF-IDF Vectorization**.
4. **Model Training**: Four machine learning models are trained on the preprocessed data:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
5. **Model Evaluation**: Each model's performance is evaluated, and you can optionally view the classification reports.
6. **Manual Testing**: A function is provided to manually input a news article and see predictions from each model.

## How to Run

1. **Install Dependencies**:
   Ensure you have the necessary Python libraries installed. You can install the required packages using the following command:

   ```bash
   pip install pandas numpy scikit-learn
   ```

2. **Prepare Dataset**:
   Download or prepare your own `True.csv` and `Fake.csv` datasets and place them in the project directory.

3. **Run the Script**:
   Execute the Python script to train the models and manually test news articles:

   ```bash
   python news_classifier.py
   ```

   During execution, you will be prompted to input a news article. Enter the text of the article you want to classify, and the models will predict whether it is **fake** or **factual**.

## Project Structure

```
├── True.csv                 # Dataset containing factual news
├── Fake.csv                 # Dataset containing fake news
├── news_classifier.py        # Main script for training and testing models
└── README.md                # Project documentation
```

## Code Explanation

1. **Data Preprocessing**: The `wordopt()` function is used to clean the text data by converting it to lowercase and removing unnecessary elements such as URLs, HTML tags, punctuation, digits, and newline characters.

2. **Model Training and Testing**: Four different machine learning models are trained on the TF-IDF transformed data. You can uncomment the `print(classification_report())` lines to see detailed performance reports for each model.

3. **Manual Testing**: The `manual_testing()` function allows you to input a news article as a string and get predictions from all the trained models.

## Example

Here is an example of how to classify a news article:

```bash
Enter the news article for classification: "The government is implementing new policies to tackle economic growth..."
```

The output will be:

```bash
LR Prediction: It's a factual news 
DTC Prediction: It's a factual news 
GBC Prediction: It's a fake news 
RFC Prediction: It's a factual news
```

## Conclusion

This project demonstrates the application of basic text preprocessing, TF-IDF vectorization, and several machine learning models to solve the problem of fake news detection. The manual testing feature allows you to experiment with real or custom news articles and observe predictions from different models.

## Future Improvements

- **Model Tuning**: Hyperparameter tuning for the models could improve their performance.
- **Deep Learning**: Exploring deep learning methods like LSTM or transformers for better accuracy.
- **Data Expansion**: Incorporating more datasets to enhance the training data and improve generalization.

Feel free to fork this project, try it with your own datasets, or modify the models for better performance!

---

**Author**: Joseph Saputra  
**Email**: josephrama1510@gmail.com
```