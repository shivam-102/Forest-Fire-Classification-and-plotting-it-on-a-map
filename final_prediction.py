import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
#df = pd.read_csv('wildfire_tweets.csv')

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]
    # Join the tokens back into text
    text = ' '.join(tokens)
    return text

df['tweet_text'] = df['tweet_text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['class_label'], test_size=0.15, random_state=42)

# Feature extraction using Bag of Words
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a random forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)

# Deploy the model to make predictions on new tweets
new_tweet = 'Alberta wildfire disrupts rescue convoy'
new_tweet_dup = new_tweet
new_tweet = preprocess_text(new_tweet)
new_tweet = vectorizer.transform([new_tweet])
prediction = model.predict(new_tweet)

print('Prediction:', prediction)

# Load the Canada map data
canada_map = gpd.read_file('canada_new_shapefile.zip')

# Filter the data
new_list = list(new_tweet_dup.split())
new_list_2 = list(canada_map['PRNAME'])
place_name=''
for i in new_list:
  for j in new_list_2:
    if(i == j):
      place_name = i
map_plot = canada_map[canada_map['PRNAME'] == place_name]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
canada_map.plot(ax=ax, alpha=0.4, color='grey')
map_plot.plot(ax=ax, color='red')
ax.set_title('Canada Map showing '+place_name)
plt.show()