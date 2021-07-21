"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
#from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

from PIL import Image
import base64





# Data Loading
title_list = load_movie_titles('movies.csv')

# App declaration
def main():

    

  

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Exploratory data analysis","Solution Overview","Company Information"]

    

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Recommender System":
        
        logo = Image.open("resources/imgs/Bestestflix.png")
        st.image(logo)
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        logo = Image.open("resources/imgs/Bestestflix.png")
        st.image(logo)
        logo1 = Image.open("resources/imgs/Image_header1.png")
        st.image(logo1)
        st.title("Solution Overview")
        st.info("Content based Filtering")
        
        st.write("Content-based filtering methods are based on a description of the item and a profile of the user's preferences.")
        st.write(" These methods are best suited to situations where there is known data on an item (name, location, description, etc.), but not on the user.")
        st.write("Content-based recommenders treat recommendation as a user-specific classification problem and learn a classifier for the user's likes and dislikes based on an item's features.")
        st.write("In this system, keywords are used to describe the items and a user profile is built to indicate the type of item this user likes.")
        st.write(" In other words, these algorithms try to recommend items that are similar to those that a user liked in the past, or is examining in the present.")
        st.write("It does not rely on a user sign-in mechanism to generate this often temporary profile. In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended. This approach has its roots in information retrieval and information filtering research.")


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


    # Building out the "Company Information, Background & Team" page



    if page_selection == "Company Information":
        logo = Image.open("resources/imgs/Bestestflix.png")
        st.image(logo)
        st.title("Company Information")
		
        st.info("Discover the mission and vision that keeps us going as well as the amazing team that pulled this project together and how we started.")    

        st.header('Our Mission')		
        st.write('To use AI & machine learning to create a world renouned recommendation system.')

        st.header('Our Vision')
        st.write('To create better and more intelligent ways to connect people to their favorite movies.')

        st.header('Our Amazing Team')
        st.write('A team of 6 passionate AI solutionists.')

        #First row of pictures

        col1, col2,col3 = st.beta_columns(3)
        Stefan_Pic =Image.open('resources/imgs/Stefan_pic.png') 
        col1.image(Stefan_Pic,caption="Stefan Ferreira", width=150)
        col1.write('CEO of Sandton')

        
        Veshen_Pic =Image.open('resources/imgs/veshen_pic.png') 
        col2.image(Veshen_Pic,caption="Veshen Naidoo", width=150)
        col2.write('CEO of Science Fiction')

        Will_Pic =Image.open('resources/imgs/Will_pic.png') 
        col3.image(Will_Pic,caption="Will Van Ieperen", width=150)
        col3.write('CEO of Comedy')

        #Second row of pictures
        col1,col2, col3 = st.beta_columns(3)
        Jason_pic =Image.open('resources/imgs/Jason_pic.png') 
        col1.image(Jason_pic,caption="Jason Alexander Dunbar", width=150)
        col1.write('CEO of Horrors')


        Ntombela_pic =Image.open('resources/imgs/Ntombela_pic.png') 
        col2.image(Ntombela_pic,caption="Bhekumuzi Ntombela", width=150)
        col2.write('CEO of Action')

        blank =Image.open('resources/imgs/blank2.png') 
        col3.image(blank, width=150)
        



        st.header('How it all began')
        st.write('We started as a group of 5 students who met each other on a university project. We bonded together around a love for solving problems with the help of AI. ')	
        st.write('We graduated with flying colours and entered successfull careers, never forgetting the joys of solving real world problems.')
        st.write('A few years later we decided to meet up and start working part time on this project which we call Bestestflix.')
	








if __name__ == '__main__':
    main()
