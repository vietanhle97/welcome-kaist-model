# how to install nltk
import nltk
# nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import OrderedDict
from operator import itemgetter
from nltk import word_tokenize
import pandas as pd
import json
import collections
import csv
# sort values in vector but preserving colums index
import os 
d = os.getcwd()

def sort(matrix):
    tuples = zip(matrix.col, matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def get_top_n(feature_names, sorted_items, topn):  
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        # store features and its score
        score_vals.append(round(score, 4))
        feature_vals.append(feature_names[idx])

    # reault (feature,score)
    results= dict()
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

def topImportant(text):
  vectorizer = CountVectorizer(max_df=0.85, stop_words = 'english')
  count_vec = vectorizer.fit_transform(text)

  # compute IDF value
  tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
  tfidf_transformer.fit(count_vec)

  features = vectorizer.get_feature_names()
  # create TF-IDF for the current text
  tf_idf_vector=tfidf_transformer.transform(vectorizer.transform(text))

  # sort the value of TF-IDF vector
  sorted_items = sort(tf_idf_vector.tocoo())

  keywords = get_top_n(features, sorted_items,10)
  # print(keywords)

  # add topic to top important word default 1 if is not exist in sortedKeywords
  topic = text['TOPIC']
  topic_tokens = word_tokenize(topic)
  if len(topic_tokens) == 2:
    first_part = topic_tokens[0]
    second_part = topic_tokens[1]
    if first_part in keywords.keys() and second_part in keywords.keys():
      pass
    else:
      if first_part in keywords.keys() and second_part != 'application' :
        keywords[second_part] = 1.0
      elif second_part in keywords.keys() and first_part != 'application':
        keywords[first_part] = 1.0
      else:
        if first_part != 'application':
          keywords[first_part] = 1.0
        if second_part != 'application':
          keywords[second_part] = 1.0

  if topic in keywords.keys():
    pass
  else:
    keywords[topic] = 1.0
  # print(keywords)
  
  sortedKeywords = OrderedDict(sorted(keywords.items(), key=itemgetter(1),reverse=True))
  # print("===============================")
  print(sortedKeywords)
  return sortedKeywords 

# res = topImportant(data.iloc[5])
# print(res['mandatory'])
# print("=====")
# data.iloc[5]
if __name__ == "__main__":
  data = pd.read_csv("data.csv")
  top_result = []
  for i in range(len(data)):
    temp = topImportant(data.iloc[i])
    top_result.append(temp)
  with open("topKeywords.json","w") as file:
    json.dump(top_result,file)


  
