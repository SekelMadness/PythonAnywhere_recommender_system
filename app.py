# Start importing relevant librairies
# Import libraries
import os
import pandas as pd
import numpy as np
from time import time
from random import randint
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k, AUC_at_k
import pickle
import flask
from flask import jsonify

clicks = pd.read_csv("https://github.com/SekelMadness/PythonAnywhere_recommender_system/releases/download/clicks/clicks.csv")
MODEL_PATH = "./recommender.model"
if not os.path.exists(MODEL_PATH):
    os.system("wget https://github.com/SekelMadness/PythonAnywhere_recommender_system/releases/download/clicks/recommender.model")
    
def compute_interaction_matrix(clicks):
    # Create interaction DF (count of interactions between users and articles)
    interactions = clicks.groupby(['user_id', 'article_id']).size().reset_index(name='count')

    # csr = compressed sparse row (good format for math operations with row slicing)
    # Create sparse matrix of shape (number_items, number_user)
    csr_item_user = csr_matrix((interactions['count'].astype(float),
                                (interactions['article_id'],
                                 interactions['user_id'])))

    # Create sparse matrix of shape (number_user, number_items)
    csr_user_item = csr_matrix((interactions['count'].astype(float),
                                (interactions['user_id'],
                                 interactions['article_id'])))

    return csr_item_user, csr_user_item

def get_cf_reco(clicks, userID, csr_item_user, csr_user_item, model_path=None, n_reco=5, train=True):
    start = time()
    # Train the model on sparse matrix of shape (number_items, number_user)

    if train or model_path is None:
        model = LogisticMatrixFactorization(factors=128, random_state=42)
        print("[INFO] : Start training model")
        model.fit(csr_user_item)

        # Save model to disk
        with open('recommender.model', 'wb') as filehandle:
            pickle.dump(model, filehandle)
    else:
        with open(MODEL_PATH, 'rb') as filehandle:
            model = pickle.load(filehandle)

    # Recommend N best items from sparse matrix of shape (number_user, number_items)
    # Implicit built-in method
    # N (int) : number of results to return
    # filter_already_liked_items (bool) : if true, don't return items present in
    # the training set that were rated/viewd by the specified user
    recommendations_list = []
    recommendations = model.recommend(userID, csr_user_item[userID], N=n_reco, filter_already_liked_items=True)

    print(f'[INFO] : Completed in {round(time() - start, 2)}s')
    print(f'[INFO] : Recommendations for user {userID}: {recommendations[0].tolist()}')

    return recommendations[0].tolist()

csr_item_user, csr_user_item = compute_interaction_matrix(clicks)

# Creating flask app
app = flask.Flask(__name__)

# This is the route to the welcome page of the recommendation API
@app.route("/")
def hello():
    return "<h1><font color= #FF7F50>Hello</font></h1><br><h2><font color= #6495ED>WELCOME THIS IS MY RECOMMENDATION API (5 BEST)</font></h2>"

# This is the route to the API
@app.route("/get_recommendation/<id>", methods=["POST", "GET"])
def get_recommendation(id):

    recommendations = get_cf_reco(clicks, int(id), csr_item_user, csr_user_item, model_path=MODEL_PATH, n_reco=5, train=False)
    data = {
            "user" : id,
            "recommendations" : recommendations,
        }
    return jsonify(data)

# # to add to the app.py file to run on local

# if __name__ == "__main__":

#     # Launch the Flask app
#     app.run(debug=False, port=8000)

