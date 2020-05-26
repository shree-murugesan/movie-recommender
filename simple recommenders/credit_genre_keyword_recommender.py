import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import numpy as np

metadata = pd.read_csv('../data/movies_metadata.csv', low_memory=False)
credits = pd.read_csv('../data/credits.csv')
keywords = pd.read_csv('../data/keywords.csv')

metadata = metadata.drop([19730, 29503, 35587]) # bad movie IDs

# for merging the dataframes
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

# return director or NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# for cast, keywords, and genres
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return [] # if there's an issue with the data

metadata['director'] = metadata['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# all lowercase and strip spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
metadata['soup'] = metadata.apply(create_soup, axis=1)

# use CountVectorizer and find cosine similarities
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

C = metadata['vote_average'].mean()
m = metadata['vote_count'].quantile(0.60)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # weighted rating formula taken from IMDB
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title] #index of given movie title

    # similarity scores of given movie with all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 'most similar' will be given movie itself
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]

# get top 25 movies according to cosine_sim score and then apply weighted rating to get top 10
def popular_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average']]

    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.60)
    # qualifying movies in a new dataframe
    qualifying_movies = movies.copy().loc[movies['vote_count'] >= m]
    qualifying_movies.shape
    qualifying_movies['score'] = qualifying_movies.apply(weighted_rating, axis=1)
    qualifying_movies = qualifying_movies.sort_values('score', ascending=False)
    return qualifying_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


print(get_recommendations('The Godfather'))
print(popular_recommendations('The Godfather'))
print(popular_recommendations('Jumanji'))