"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from scipy.sparse import csr_matrix
import scipy as sp

# Packages for saving models
import pickle

df_movie_imdb_tags = pd.read_csv("main_df_3.csv")


# Convienient indexes to map between book titles and indexes of 
# the movies dataframe
titles = df_movie_imdb_tags['title']
indices = pd.Series(df_movie_imdb_tags.index, index=df_movie_imdb_tags['title'])

tf = TfidfVectorizer(min_df = 10)

 # Produce a feature matrix, where each row corresponds to a movie,
# with TF-IDF features as columns 
tf_comb_matrix = tf.fit_transform(df_movie_imdb_tags['combined_features'])

#cosine_sim_comb = np.load('cosine_sim_comb_3.npy')
cosine_sim_comb = cosine_similarity(tf_comb_matrix,tf_comb_matrix)

def content_generate_top_N_recommendations_list(movie_title, N=10):
    N = N+1
    # Place 'The' at the end
    if movie_title.startswith('The'):
        movie_title = movie_title[4:-7] + ', The ' + movie_title[-6:]
    
    # Convert the string movie title to a numeric index for our 
    # similarity matrix
    b_idx = indices[movie_title]
    # Extract all similarity values computed with the reference book title
    sim_scores = list(enumerate(cosine_sim_comb[b_idx]))
    # Sort the values, keeping a copy of the original index of each value
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Select the top-N values for recommendation
    sim_scores = sim_scores[1:N]
    # Collect indexes 
    movie_indices = [i[0] for i in sim_scores]
    # Convert the indexes back into titles 
    movies = list(titles.iloc[movie_indices].values)

    # Return list of top 10 movies with "The" at the start of the movie title
    fixed_list = []
    for movie in movies:
        if movie[:-7].endswith('The'):
            fixed_list.append('The ' + movie[:-12] + ' ' + movie[-6:])
        elif movie[:-7].endswith('A'):
            fixed_list.append('A ' + movie[:-10] + ' ' + movie[-6:])
        else:
            fixed_list.append(movie)
    
    return(fixed_list[:N])


  
# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):

    rec1 = content_generate_top_N_recommendations_list(movie_list[0], N=10)
    rec2 = content_generate_top_N_recommendations_list(movie_list[1], N=10)
    rec3 = content_generate_top_N_recommendations_list(movie_list[2], N=10)

    final_list = rec1[0:4] + rec2[0:3] + rec3[0:3]

    return final_list
    