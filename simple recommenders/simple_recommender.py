import pandas as pd

metadata = pd.read_csv('../data/movies_metadata.csv', low_memory=False)

C = metadata['vote_average'].mean()
m = metadata['vote_count'].quantile(0.90)
# qualifying movies in a new dataframe
qualifying_movies = metadata.copy().loc[metadata['vote_count'] >= m]
qualifying_movies.shape

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # weighted rating formula taken from IMDB
    return (v/(v+m) * R) + (m/(m+v) * C)

qualifying_movies['score'] = qualifying_movies.apply(weighted_rating, axis=1)
qualifying_movies = qualifying_movies.sort_values('score', ascending=False)

print(qualifying_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))