import pandas as pd
import numpy as np
import syllables
from nltk.stem import WordNetLemmatizer
from nltk import tokenize,ngrams
import nltk
from nltk.corpus import stopwords, wordnet
import re
import textstat
import itertools
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
from collections import Counter 
from textblob import TextBlob

df = pd.read_excel('Movie reviews.xlsx')

df['Words'] = df['Review'].apply(lambda x : len( x.split()) )
df['Total syllables'] = df['Review'].apply(lambda x : syllables.estimate(x))
df['flesch reading ease'] = df['Review'].apply(lambda x : textstat.flesch_reading_ease(x) )

# extract tokens (words) without whitespaces, new line and tabs
w_tokenizer = tokenize.WhitespaceTokenizer()

# transform tokens into lemmas
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower()) if w not in stopwords.words('english')]

df['Lemmas'] = df['Review'].apply(lambda x : lemmatize_text(x))

# hue is splitting bar into several bar categories per selected column
ax = sns.countplot(x='Year', hue = "State", data = df, palette=['#10c456',"#fb350d"]).set_title("No. of positive and negative reviews per year")
plt.savefig("assets/chart1.png")

rotten_before_nom = len( df[(df["State"] == "rotten") & (df["Date"] < '2024-01-23')] )
rotten_after_nom = len( df[(df["State"] == "rotten") & (df["Date"] >= '2024-01-23')] )

# print("Count of negative reviews published before oscar nomination:")
# print(rotten_before_nom)
# print("Count of negative reviews published after oscar nomination:")
# print(rotten_after_nom)

# WC for unigrams
for rating in ["rotten", "fresh"]:
    curr_lemmatized_tokens = list(df[df['State'] == rating]['Lemmas'])
    # Convert a collection of text documents to a matrix of token counts:
    # The lower and upper boundary of the range for different word n-grams
    # (2, 2) means only bigrams
    vectorizer = CountVectorizer(ngram_range=(1,1))
    bag_of_words = vectorizer.fit_transform(df[df['State'] == rating]['Lemmas'].apply(lambda x : ' '.join(x)))
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 30
    wordCloud = WordCloud(max_words = WC_max_words, height = WC_height, width = WC_width)
    
    wordCloud.generate_from_frequencies(words_dict)
    plt.figure(figsize=(10,8))
    plt.imshow(wordCloud)
    if rating == "rotten":
        state = "negative"
    else:
        state = "positive"
    plt.title('Word Cloud for ' + str(state) + ' reviews', fontsize = 30)
    plt.axis("off")
    plt.savefig('assets/Word Cloud for ' + str(state) + ' reviews.png')
    plt.show()

# WC for bigrams
for rating in ["rotten", "fresh"]:
    curr_lemmatized_tokens = list(df[df['State'] == rating]['Lemmas'])
    vectorizer = CountVectorizer(ngram_range=(2,2))
    bag_of_words = vectorizer.fit_transform(df[df['State'] == rating]['Lemmas'].apply(lambda x : ' '.join(x)))
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 30
    wordCloud = WordCloud(max_words = WC_max_words, height = WC_height, width = WC_width, colormap='Oranges')
    
    wordCloud.generate_from_frequencies(words_dict)
    plt.figure(figsize=(10,8))
    plt.imshow(wordCloud)
    if rating == "rotten":
        state = "negative"
    else:
        state = "positive"
    plt.title('Word Cloud of bigrams for ' + str(state) + ' reviews', fontsize = 25)
    plt.axis("off")
    plt.savefig('assets/Word Cloud of bigrams for ' + str(state) + ' reviews.png')
    plt.show()

lemmatized_tokens = list(df['Lemmas'])
vectorizer = CountVectorizer(ngram_range=(1,1))
bag_of_words = vectorizer.fit_transform(df['Lemmas'].apply(lambda x : ' '.join(x)))
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
sorted_words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
word_freq_counter = Counter({word: freq for word, freq in sorted_words_freq})

most_occur = word_freq_counter.most_common(10) 

words = [x[0] for x in most_occur]
counts = [x[1] for x in most_occur]

plt.barh(words, counts, color='skyblue')
plt.title("Most common words")
plt.gca().invert_yaxis()
plt.savefig("assets/chart6.png")
plt.show()

plt.figure()
plt.hist(df['flesch reading ease'], bins = 10,  color='skyblue')
plt.xlabel("flesch reading ease")
plt.ylabel("frequency")
plt.title("Distribution of flesch reading ease")
plt.savefig("assets/chart7.png")
plt.show()

def calculate_polarity(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Polarity'] = df['Review'].apply(calculate_polarity)
polarity_counts = df['Polarity'].value_counts()

comparison_df = df[['State', 'Polarity']].copy()

comparison_df = comparison_df.replace({'State': {'rotten': 'Negative', 'fresh': 'Positive'}})
comparison_df['Match'] = comparison_df['State'] == comparison_df['Polarity']

value_counts = comparison_df['Match'].value_counts()

plt.figure(figsize=(8, 6))
value_counts.plot(kind='bar', color='skyblue')
plt.title('How good TextBlob is matching polarity of reviews?', fontsize = 16)
plt.xlabel('Match')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig("assets/chart8.png")
plt.show()

total_reviews = len(df['Review'])
negative_reviews = df['State'].value_counts()['rotten']
positive_reviews = df['State'].value_counts()['fresh']
