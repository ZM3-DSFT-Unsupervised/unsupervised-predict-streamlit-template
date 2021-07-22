"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df_train = pd.read_csv("cust_train_df.csv")
df_test = pd.read_csv("cust_test_df.csv")
df_movies = pd.read_csv("movies.csv")
loaded_model=pickle.load(open('resources/models/quick_svd.pkl', 'rb'))
  
# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    
    movie_id1 = df_movies.loc[df_movies["title"].isin(movie_list), "movieId"].iloc[0]

    user_IDs = (df_train[df_train['movieId']==movie_id1].sort_values(by = 'rating', ascending = False)).userId[:50].values

    combo_df = df_test[df_test['userId'].isin(user_IDs)]

    predict_colb = []
    for i, row in combo_df.iterrows():
       x = (loaded_model.predict(row.userId, row.movieId))
       pred_colab = x[3]
       predict_colb.append(pred_colab)

    results = pd.DataFrame({"userId":combo_df['userId'],"movieId":combo_df['movieId'],"rating": predict_colb})
    results = results.sort_values(by = 'rating', ascending= False)
    results = pd.merge(results, df_movies, on='movieId', how = 'left')

    colab_list = list(results.title[:100])

    top_10 = list(dict.fromkeys(colab_list))

    return top_10[:10]
