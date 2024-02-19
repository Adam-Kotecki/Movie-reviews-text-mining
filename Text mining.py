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
ax = sns.countplot(x='Year', hue = "State", data = df, palette=['#10c456',"#fb350d"])

rotten_before_nom = len( df[(df["State"] == "rotten") & (df["Date"] < '2024-01-23')] )
rotten_after_nom = len( df[(df["State"] == "rotten") & (df["Date"] >= '2024-01-23')] )

print("Count of negative reviews published before oscar nomination")
print(rotten_before_nom)
print("Count of negative reviews published after oscar nomination")
print(rotten_after_nom)


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
    plt.figure(figsize=(20,8))
    plt.imshow(wordCloud)
    if rating == "rotten":
        state = "negative"
    else:
        state = "positive"
    plt.title('Word Cloud for ' + str(state) + ' reviews', fontsize = 30)
    plt.axis("off")
    plt.show()


'''
print(len(df[df["State"] == "fresh"]))


labels = []
# containers are groups of bars
for container in ax.containers:
    group_height = 0
    for bar in container:
        # getting height of current bar
        bar_height = bar.get_height()
        group_height =+ bar_height
    for bar in container:
        bar_height = bar.get_height()
        labels.append(f'{group_height/bar_height*100:0.1f}%')
        
            
    #labels = [f'{h/df.State.count()*100:0.1f}%' if (h := v.get_height()) > 0 else '' for v in c]
ax.bar_label(container, labels = labels, label_type='edge')
'''
'''    
rotten = df[df["State"] == "rotten"]
fresh = df[df["State"] == "fresh"]

perc = df['State'].value_counts(ascending=False, normalize=True).values * 100
percentage = pd.DataFrame({'count': [fresh.shape[0], rotten.shape[0]], '%': perc }, index = ['Fresh', 'Rotten'])
print(percentage)



labels = []
# containers are groups of bars
for c in ax.containers:
    # b is single bar, described by width and height
    group_height = 0
    for b in enumerate(c):
        # getting height of current bar
        h = b[1].get_height()
        group_height =+ h
        # checking index of category:
        if b[0] == 0:
            labels.append(f'{group_height/df.State["fresh"].count()*100:0.1f}%')
        else:
            labels.append(f'{group_height/df.State["rotten"].count()*100:0.1f}%')
'''



# Extracting emotions words:

def get_emotion_words(reviews):
    emotion_words = set()
    for review in reviews:
        words = nltk.word_tokenize(review)
        tagged_words = nltk.pos_tag(words)
        for word, tag in tagged_words:
            # checking if tag is adjective:
            if tag.startswith('JJ'):  # Adjectives
                synonyms = set()
                # Synset instances are the groupings of synonymous words that express the same concept
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                emotion_words.update(synonyms)
    return emotion_words

pos_reviews = list(df.loc[df['State'] == "fresh", 'Review'])
neg_reviews = list(df.loc[df['State'] == "rotten", 'Review'])

pos_emotions = get_emotion_words(pos_reviews)
neg_emotions = get_emotion_words(neg_reviews)

print(pos_emotions)
