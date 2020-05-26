import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)
reader = Reader()
ratings = pd.read_csv('data/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
# cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)
# predict rating for user_id = 1 for movie_id = 302
print(svd.predict(1, 302, 3))