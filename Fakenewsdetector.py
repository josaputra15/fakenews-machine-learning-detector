# Importing necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the datasets for true and fake news
true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

# Label the datasets: 1 for true news, 0 for fake news
true['label'] = 1
fake['label'] = 0

# Concatenate the two datasets into one
news = pd.concat([fake, true], axis=0)

# Check for missing values in the dataset
news.isnull().sum()

# Drop unnecessary columns (title, subject, and date) from the dataset
news = news.drop(['title', 'subject', 'date'], axis=1)

# Shuffle the dataset to ensure a random distribution
news = news.sample(frac=1).reset_index(drop=True)

# Function to preprocess text data (cleaning and normalizing)
def wordopt(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d', '', text)
    # Remove newline characters
    text = re.sub(r'\n', '', text)
    return text

# Apply text preprocessing to the 'text' column in the dataset
news['text'] = news['text'].apply(wordopt)

# Separate features (text) and labels (true/fake)
x = news['text']
y = news['label']

# Split the data into training and testing sets (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Initialize the TF-IDF Vectorizer to convert text into numerical features
vectorization = TfidfVectorizer()

# Fit and transform the training data and transform the test data
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression Model
LR = LogisticRegression()
LR.fit(xv_train, y_train)  # Train the model
pred_lr = LR.predict(xv_test)  # Predict on the test set
# print(classification_report(y_test, pred_lr))  # Classification report (optional)

# Decision Tree Classifier Model
DTC = DecisionTreeClassifier()
DTC.fit(xv_train, y_train)  # Train the model
pred_dtc = DTC.predict(xv_test)  # Predict on the test set
# print(classification_report(y_test, pred_dtc))  # Classification report (optional)

# Random Forest Classifier Model
RFC = RandomForestClassifier()
RFC.fit(xv_train, y_train)  # Train the model
pred_rfc = RFC.predict(xv_test)  # Predict on the test set
# print(classification_report(y_test, pred_rfc))  # Classification report (optional)

# Gradient Boosting Classifier Model
GBC = GradientBoostingClassifier()
GBC.fit(xv_train, y_train)  # Train the model
pred_gbc = GBC.predict(xv_test)  # Predict on the test set
# print(classification_report(y_test, pred_gbc))  # Classification report (optional)

# Function to interpret the prediction result (label) and output the corresponding message
def output_label(n):
    if n == 0:
        return "It's a fake news"
    elif n == 1:
        return "It's a factual news"

# Manual testing function to classify a single news article
def manual_testing(news):
    # Create a DataFrame with the input news text
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    
    # Preprocess the text input using the same wordopt function
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    
    # Transform the input text into numerical features using the TF-IDF vectorizer
    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)
    
    # Predict using the different models
    pred_lr = LR.predict(new_xv_test)
    pred_dtc = DTC.predict(new_xv_test)
    pred_gbc = GBC.predict(new_xv_test)
    pred_rfc = RFC.predict(new_xv_test)
    
    # Return the predictions from each model in a readable format
    return "\n\nLR Prediction: {} \nDTC Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_label(pred_lr[0]), output_label(pred_dtc[0]), output_label(pred_gbc[0]), output_label(pred_rfc[0])
    )

# Input: News article for manual testing
news_article = str(input("Enter the news article for classification: "))

# Call the manual testing function and display predictions
print(manual_testing(news_article))
