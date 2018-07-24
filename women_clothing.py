import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Womens_Clothing_Reviews.csv',header=0)
data = data.rename(columns={list(data)[0]: 'count'})
data.head()
data.describe()

data.groupby(['Department Name', 'Rating']).count()

sns.countplot(x='Department Name', hue='Rating', data=data)

sns.countplot(x='Rating', hue='Recommended IND', data=data)

sns.countplot(x='Department Name', hue='Recommended IND',data=data)

# set the text to string
data['Review Text'] = data['Review Text'].astype(str)
# how many character in a string
data['Text_len'] = data['Review Text'].apply(len)

np.average(data['Review Text'].apply(len))

# average text length by rating
data.groupby(['Rating'])['Text_len'].apply(np.average)


# rating 1 and 5 have the lowest average text length, while rating of 3 has the highest average text length
# one of the explanations is that people who rated 3 actually explained in detail why they rated as 3.
plt.bar(x=[1,2,3,4,5], height=data.groupby(['Rating'])['Text_len'].apply(np.average))
sns.boxplot(x='Rating', y='Text_len',data=data)

# the average age of each Rating group is close to each other.
sns.boxplot(x='Rating', y='Age', data=data)

import nltk
# use nltk.download() to download all packages if needed

# tokenizing text
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# implementing stop words
sw = stopwords.words('english')
# Stemming means removing affixes (suffix/ prefix) from words and returning the root word. (detailed explanation below)
stemmer = PorterStemmer()
text = " ".join(data['Review Text'])
tokens = word_tokenize(text)
new_tokens = []
for word in tokens:
    if len(word) <= 2 or word in sw:
        continue
    new_tokens.append(stemmer.stem(word))


# counting word frequency
# plotting the frequency of top 20 words
freq = nltk.FreqDist(new_tokens)
freq.plot(20)

# implementing sentiment analysis
# SentimentIntensityAnalyzer's output has 4 scores:
# neg: Negative, neu: Neutral, pos: Positive, and compound: aggregated score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
pos = 0
neg = 0
neu = 0
m1 = SentimentIntensityAnalyzer()
for sentence in data['Review Text']:
    pos += m1.polarity_scores(sentence)['pos']
    neg += m1.polarity_scores(sentence)['neg']
    neu += m1.polarity_scores(sentence)['neu']

# alternative way: select the max score
pos = 0
neg = 0
neu = 0
for sentence in data['Review Text']:
    if m1.polarity_scores(sentence)['neu'] >= m1.polarity_scores(sentence)['neg'] and m1.polarity_scores(sentence)['neu'] >= m1.polarity_scores(sentence)['pos']:
        neu += 1
    elif m1.polarity_scores(sentence)['pos'] == m1.polarity_scores(sentence)['neg']:
        neu += 1
    elif m1.polarity_scores(sentence)['pos'] > m1.polarity_scores(sentence)['neg']:
        pos += 1
    elif m1.polarity_scores(sentence)['neg'] > m1.polarity_scores(sentence)['pos']:
        neg += 1

pos
neg
neu
sum([pos,neg,neu])






# --------------------------------------
# Stemming vs Lemmatizing
# Stemming means removing affixes (suffix/ prefix) from words and returning the root word.
# However, Stemming might delete some characters that were not suppose to, ex. increases
# Lemmatizing can solve such issue:
# Stemming:
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# print(stemmer.stem('working'))
# lemmatizing words
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
lem.lemmatize('this')




