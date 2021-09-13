# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:29:08 2020

@author: user
"""



import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity




## getting the dataset 
df = pd.read_csv('movie_dataset.csv')


## the features that will take the similarity with 
features = ['keywords','cast','genres','director']

## make features defult  
df[features]=df[features].fillna('')




## function for combine the row of each features   
def featurescombine(row):

		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]







## take the combine features with agg to sort the features on axis
df["featurescombine"]=df.agg(featurescombine,axis=1)




##count matrix of the features
vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform(df["featurescombine"])
## get the similarity of the matrix
similarity=cosine_similarity(matrix)



##function that take the title to get its index 
def get_index(title):

	return df[df.title == title]["index"].values[0]

## the input 
movie = "Batman"
## get index of this movie 
movie_index = get_index(movie)



## take the index list of the similar input of this index
similar_movies =  list(enumerate(similarity[movie_index]))




## sort the similar of the inputs
sorted_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


##get title of the index 
def get_title(index):

	return df[df.index == index]["title"].values[0]
i=0
## getting the movies that similar to the user movie
for item in sorted_movies:

		print (get_title(item[0]))

		i=i+1

		if i>5:

			break

