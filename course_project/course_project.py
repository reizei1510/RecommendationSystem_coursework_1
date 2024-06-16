# -*- coding: cp1251 -*-

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

import pandas as pd
import string
import pymorphy2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('articles.csv', ';', names=['ID', 'Title', 'Text'], encoding='cp1251')[1:]

df['Tokens'] = ''
for index, row in df.iterrows():
    row['Tokens'] = word_tokenize(row['Text'], language="russian")

df['Key words'] = ''
morph = pymorphy2.MorphAnalyzer()
for index, row in df.iterrows():
    for word in row['Tokens']:
        if (word not in stopwords.words("russian")) and (word not in string.punctuation):
            row['Key words'] += morph.parse(word)[0].normal_form + " "

#for index, row in df.iterrows():
    #key_words = pd.DataFrame([[row['ID'], row['Title'], row['Key words']]], columns=['ID', 'Title', 'Key words'])
    #key_words.to_csv('key_words.csv', mode='a', sep=';', header=False, encoding='cp1251', index=False)

count = CountVectorizer()
count_matrix = count.fit_transform(df['Key words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def recommend(id, cosine_sim):
    recommended_articles = []
    score_series = pd.Series(cosine_sim[id-1]).sort_values(ascending = False)
    top_5_indices = list(score_series.iloc[1:6].index)
    
    for i in top_5_indices:
        recommended_articles.append(list(df['Title'])[i])
        
    return recommended_articles

id = 3
recommended_articles = recommend(id, cosine_sim)
print("\n\nСтатьи, похожие на \"" + pd.Series(df["Title"] + "\":\n")[id])
for article in recommended_articles:
    print(article)
print("\n\n")
