#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
import nltk
import collections
from collections import Counter
from nltk.stem.porter import *  
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import string


# In[2]:


words_to_remove = stopwords.words('english')
puncs = list(string.punctuation)
words_to_remove.extend(puncs)
words_to_remove.extend(['``','""',"''","'s","...","-","_","–"])
def get_tokens(file_path):
    with open(file_path, 'r') as shakes:
        text = shakes.read()
        tokens = nltk.word_tokenize(text)
    return tokens
tokens1 = get_tokens('MIT.txt')
tokens2 = get_tokens('UIC.txt')
tokens3 = get_tokens('TESLA.txt')
tokens4 = get_tokens('STANFORD.txt')
tokens5 = get_tokens('UIS.txt')
tokens6 = get_tokens('UIUC.txt')
filtered_tokens1 = [w.lower() for w in tokens1 if not w.lower() in words_to_remove]
filtered_tokens2 = [w.lower() for w in tokens2 if not w.lower() in words_to_remove]
filtered_tokens3 = [w.lower() for w in tokens3 if not w.lower() in words_to_remove]
filtered_tokens4 = [w.lower() for w in tokens4 if not w.lower() in words_to_remove]
filtered_tokens5 = [w.lower() for w in tokens5 if not w.lower() in words_to_remove]
filtered_tokens6 = [w.lower() for w in tokens6 if not w.lower() in words_to_remove]

count1 = Counter(filtered_tokens1)
print(count1.most_common(100))
count2 = Counter(filtered_tokens2)
print(count2.most_common(100))
count3 = Counter(filtered_tokens3)
print(count3.most_common(100))
count4 = Counter(filtered_tokens4)
print(count4.most_common(100))
count5 = Counter(filtered_tokens5)
print(count5.most_common(100))
count6 = Counter(filtered_tokens6)
print(count6.most_common(100))


# In[3]:


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()
stemmed1 = stem_tokens(filtered_tokens1,stemmer)
stemmed2 = stem_tokens(filtered_tokens2,stemmer)
stemmed3 = stem_tokens(filtered_tokens3,stemmer)
stemmed4 = stem_tokens(filtered_tokens4,stemmer)
stemmed5 = stem_tokens(filtered_tokens5,stemmer)
stemmed6 = stem_tokens(filtered_tokens6,stemmer)
count1 = Counter(stemmed1)
print(count1.most_common(100))
count2 = Counter(stemmed2)
print(count2.most_common(100))
count3 = Counter(stemmed3)
print(count3.most_common(100))
count4 = Counter(stemmed4)
print(count4.most_common(100))
count5 = Counter(stemmed5)
print(count5.most_common(100))
count6 = Counter(stemmed6)
print(count6.most_common(100))


# In[4]:


def get_jaccard_sim(doc1,doc2): 
    d1 = set(doc1)
    d2 = set(doc2)
    d1_intersection_d2 = d1.intersection(d2)
    return float(len(d1_intersection_d2)) / (len(d1) + len(d2) - len(d1_intersection_d2))
jackardia_similarity21 = get_jaccard_sim(stemmed1,stemmed2)
print("Jaccardian similarity between UIC and MIT -->", jackardia_similarity21)
jackardia_similarity23 = get_jaccard_sim(stemmed3,stemmed2)
print("Jaccardian similarity between UIC and TESLA -->",jackardia_similarity23)
jackardia_similarity24 = get_jaccard_sim(stemmed4,stemmed2)
print("Jaccardian similarity between UIC and STANFORD -->",jackardia_similarity24)
jackardia_similarity25 = get_jaccard_sim(stemmed5,stemmed2)
print("Jaccardian similarity between UIC and UIS -->",jackardia_similarity25)
jackardia_similarity26 = get_jaccard_sim(stemmed6,stemmed2)
print("Jaccardian similarity between UIC and UIUC -->",jackardia_similarity26)


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
def get_tfidf(document1,document2):
    doc1 = " ".join(document1)
    doc2 = " ".join(document2)
    total_doc = [doc1,doc2]
    vectorizer = TfidfVectorizer()
    fitted_tfidf = vectorizer.fit(total_doc)
    doc1_tfidf = fitted_tfidf.transform([doc1])
    doc2_tfidf = fitted_tfidf.transform([doc2])
    return doc1_tfidf, doc2_tfidf

doc1_tfidf, doc21_tfidf = get_tfidf(stemmed1,stemmed2)
print("TFIDF for UIC and MIT \n",doc1_tfidf.data)
doc3_tfidf, doc23_tfidf = get_tfidf(stemmed3,stemmed2)
print("TFIDF for UIC and TESLA \n",doc3_tfidf.data)
doc4_tfidf, doc24_tfidf = get_tfidf(stemmed4,stemmed2)
print("TFIDF for UIC and STANFORD \n",doc4_tfidf.data)
doc5_tfidf, doc25_tfidf = get_tfidf(stemmed5,stemmed2)
print("TFIDF for UIC and UIS \n",doc5_tfidf.data)
doc6_tfidf, doc26_tfidf = get_tfidf(stemmed6,stemmed2)
print("TFIDF for UIC and UIUC \n",doc6_tfidf.data)


# In[6]:


from sklearn.metrics.pairwise import cosine_similarity
doc_cosine_similarity = cosine_similarity(doc1_tfidf, doc21_tfidf)
print("Cosine similarity between UIC and MIT -->",doc_cosine_similarity)
doc_cosine_similarity = cosine_similarity(doc3_tfidf, doc23_tfidf)
print("Cosine similarity between UIC and TESLA -->",doc_cosine_similarity)
doc_cosine_similarity = cosine_similarity(doc4_tfidf, doc24_tfidf)
print("Cosine similarity between UIC and STANFORD -->",doc_cosine_similarity)
doc_cosine_similarity = cosine_similarity(doc5_tfidf, doc25_tfidf)
print("Cosine similarity between UIC and UIS -->",doc_cosine_similarity)
doc_cosine_similarity = cosine_similarity(doc6_tfidf, doc26_tfidf)
print("Cosine similarity between UIC and UIUC -->",doc_cosine_similarity)

