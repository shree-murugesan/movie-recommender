import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('../data/movies_metadata.csv', low_memory=False)
metadata['overview'] = metadata['overview'].fillna('') # replace NaN with ''

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['overview'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# reverse map with movie indices and titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

C = metadata['vote_average'].mean()
m = metadata['vote_count'].quantile(0.90)

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
    m = movies['vote_count'].quantile(0.90)
    # qualifying movies in a new dataframe
    qualifying_movies = movies.copy().loc[movies['vote_count'] >= m]
    qualifying_movies.shape
    qualifying_movies['score'] = qualifying_movies.apply(weighted_rating, axis=1)
    qualifying_movies = qualifying_movies.sort_values('score', ascending=False)
    return qualifying_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


print(get_recommendations('The Godfather'))
print(popular_recommendations('The Godfather'))