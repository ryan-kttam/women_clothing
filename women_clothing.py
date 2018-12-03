import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer # Term Frequency times inverse document freq
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# loading Data
data = pd.read_csv('C:/Github/women_clothing/Womens_Clothing_Reviews.csv', header=0)
data = data.rename(columns={list(data)[0]: 'count'})
len(data) # there are 23486 reviews
data.head(2)
list(data)
# Data exploration,

# te most popular clothing department is "Top", followed by "Dresses"
# This company did a decent job on keeping customers rating, as Rating counts decrease gradually as rating decrease
plot1 = sns.countplot(x='Department Name', hue='Rating', data=data)
plot1.set_title('Review Rating Distribution',size=15)
plt.show(plot1)

sns.countplot(x='Rating', hue='Recommended IND', data=data)
sns.countplot(x='Department Name', hue='Recommended IND', data=data)

# TRY TO IGNORE ALL OTHER FEATURE EXCEPT rating and texts

# Customer review (text) analysis
# Setting the text to string
data['Review Text'] = data['Review Text'].astype(str)
# storing how many character in a string
data['Text_len'] = data['Review Text'].apply(len)

# average text length by rating
# rating 1 and 5 have the lowest average text length, while rating of 3 has the highest average text length
# one of the explanations is that people who rated 3 actually explained in detail why they rated as 3.
plot1 = sns.boxplot(x='Rating', y='Text_len', data=data)
plot1.set_ylabel('Average Review length')
plot1.set_title('Average Review length by Rating', size = 15)
plt.show(plot1)
# use nltk.download() to download all packages if needed

wc = WordCloud(background_color='white').generate(data['Review Text'][2])
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show(wc)

def join_all_reviews(s):
    return ' '.join(i for i in s)

list_of_word_by_rating = data.groupby(by='Rating')['Review Text'].apply(join_all_reviews)

sw = set(STOPWORDS)
sw.update(['dress', 'top', 'shirt'])
def generate_word_cloud(rating):
    wc = WordCloud(stopwords=sw, background_color='white').generate(list_of_word_by_rating[rating])
    plt.imshow(wc, interpolation='bilinear')
    plt.title('Rating: '+ str(rating), size=20)
    plt.axis('off')
    plt.show(wc)

generate_word_cloud(5)

# ----------------------------------- remove
# tokenizing text

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

pos = 0
neg = 0
neu = 0
m1 = SentimentIntensityAnalyzer()
m1.polarity_scores('i love you')
for sentence in data['Review Text']:
pos += m1.polarity_scores(sentence)['pos']
neg += m1.polarity_scores(sentence)['neg']
neu += m1.polarity_scores(sentence)['neu']
# ------------------------------------------------

m1 = SentimentIntensityAnalyzer()
sentiment = {}
for rating, rating_text in data.groupby('Rating')['Review Text']:
    pos = 0
    neg = 0
    neu = 0
    for i in rating_text:
        pos += m1.polarity_scores(i)['pos']
        neg += m1.polarity_scores(i)['neg']
        neu += m1.polarity_scores(i)['neu']
    rating_length = len(rating_text)
    result = [neg, neu, pos]
    sentiment[rating] = list(map(lambda x: x/rating_length, result))

sentiment.keys()
neg_values = [i[0] for i in sentiment.values()]
neu_values = [i[1] for i in sentiment.values()]
pos_values = [i[2] for i in sentiment.values()]

plot_neg = plt.bar(sentiment.keys(), neg_values )
plot_neu = plt.bar(sentiment.keys(), neu_values, bottom=neg_values)
plot_pos = plt.bar(sentiment.keys(), pos_values, bottom=[a+b for a, b in zip(neg_values, neu_values)])
plt.legend((plot_neg[0], plot_neu[0], plot_pos[0]), ('Negative', 'Neutral','Positive'), loc='right')
plt.ylabel('Percentage')
plt.xlabel('Rating')
plt.title('Sentiment Percentage by Rating', size=15)
plt.show()

# https://medium.com/…/more-nlp-with-sklearns-countvectorizer…
# http://adataanalyst.com/sc…/countvectorizer-sklearn-example/

import string
def clean_mess(text):
    step1 = [i for i in text if i not in string.punctuation]
    step2 = ''.join(step1)
    step3 = [i for i in step2.split() if i.lower() not in sw]
    return step3
a = ['seuwoenfe. wenruiw', 'We just hung onto each other. You couldnt even stand, Sheila Bailey']
clean_mess(a[1])

vector = CountVectorizer(stop_words=sw, ngram_range=(1,3),
                         min_df=9,
                         analyzer=clean_mess) # can apply stopword, tokenizer etc.
vector2 = TfidfTransformer()
training = vector.fit_transform(data['Review Text'])
training2 = vector2.fit_transform(training)

train_x, test_x, train_y, test_y = train_test_split(training2, data['Rating'], test_size=0.25)

def performance (train_x, test_x,train_y, test_y, model):
    actual = list(test_y)
    train_model = model.fit(train_x, train_y)
    prediction = train_model.predict(test_x)
    i = 0
    result = 0
    while i < len(actual):
        if prediction[i] == actual[i]:
            result += 1
        i += 1
    print('prediction Accuracy is: ', np.round(result * 100.0 / len(prediction), 2), '%')

# Randomforest performed slightly better than SVM and Naive Bayes.

print('Accuracy of predicting Rating is: ', np.round(result*100.0 / len(prediction), 2), '%')

performance(train_x, test_x, train_y, test_y, MultinomialNB())
performance(train_x, test_x, train_y, test_y, SVC())
performance(train_x, test_x, train_y, test_y, RandomForestClassifier(n_estimators=200))

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

