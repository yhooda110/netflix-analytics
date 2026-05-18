import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommendation_system(df):

    tfidf = TfidfVectorizer(stop_words='english')

    df['description'] = df['description'].fillna('')

    tfidf_matrix = tfidf.fit_transform(df['description'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return cosine_sim, indices


def get_recommendations(title, cosine_sim, indices, df):

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores,
                        key=lambda x: x[1],
                        reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]
