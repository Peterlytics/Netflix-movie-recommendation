#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing the necessary libaries

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


# In[7]:


#importing the csv file 

df = pd.read_csv(r"C:\Users\User2\Downloads\netflixData.csv")


# In[8]:


#to view the first five rows

df.head()


# In[10]:


#descriptive statistic of the dataset

df.describe()


# In[11]:


df.isnull().sum()


# In[13]:


df = df[["Title", "Description", "Content Type", "Genres"]]


# In[14]:


df.head()


# In[15]:


df = df.dropna()


# In[16]:


df.head(1000)


# In[17]:


df.isnull().sum()


# In[19]:


import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


# In[20]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["Title"] = df["Title"].apply(clean)


# In[34]:


print(df.Title.sample(10))


# In[35]:


feature = df["Genres"].tolist()
tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)


# In[36]:


indices = pd.Series(df.index, 
                    index=df['Title']).drop_duplicates()


# In[39]:


def netFlix_recommendation(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return df['Title'].iloc[movieindices]


# In[40]:


print(netFlix_recommendation("girlfriend"))


# In[42]:


print(netFlix_recommendation("super monster monster pet"))


# In[43]:


print(netFlix_recommendation("bal ganesh"))

