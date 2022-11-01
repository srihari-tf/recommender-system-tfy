import streamlit as st
import mlfoundry
import pandas as pd
from scipy.sparse import coo_matrix
import implicit
import os

@st.cache(allow_output_mutation=True)
def load_models_and_dfs():
  client = mlfoundry.get_client(api_key=os.getenv('TFY_API_KEY'))
  run = client.get_run(os.getenv('MLF_RUN_FQN'))

  movies_local_path = run.download_artifact('movies_metadata.csv')
  ratings_local_path = run.download_artifact('ratings_small.csv')

  movie_meta_df = pd.read_csv(movies_local_path)
  ratings_df = pd.read_csv(ratings_local_path)

  # only keep movies in ratings dataset
  movie_meta_df = movie_meta_df[movie_meta_df['id'].isin(ratings_df['movieId'].astype('string'))]

  ratings_df['movieId'] = ratings_df['movieId'].astype("category")
  ratings_df['userId'] = ratings_df['userId'].astype("category")
  r = coo_matrix((ratings_df['rating'], (ratings_df['userId'].cat.codes, ratings_df['movieId'].cat.codes)))

  user_category_to_code = dict([(category, code) for code, category in enumerate(ratings_df.userId.cat.categories)])
  movie_category_to_code = dict([(category, code) for code, category in enumerate(ratings_df.movieId.cat.categories)])

  model_local_path = run.download_artifact('recommendation-model.npz')
  model = implicit.als.AlternatingLeastSquares(factors=25).load(model_local_path)

  return model, user_category_to_code, movie_category_to_code, movie_meta_df, ratings_df, r


model, user_category_to_code, movie_category_to_code, movie_meta_df, ratings_df, r = load_models_and_dfs()

def search_movie(name):
  return (movie_meta_df.loc[movie_meta_df['original_title'].str.contains(name, case=False)][['original_title', 'id']]).to_dict('records')

def find_similar_movie(movie_name):
  search_result =search_movie(movie_name)
  if len(search_result) > 0:
    movie_id = search_result[0]['id']
    movie_name = search_result[0]['original_title']
  else:
    return []
  movie_cat_code = movie_category_to_code[int(movie_id)]
  movie_cat_codes = model.similar_items(movie_cat_code)[0]
  ids = [ratings_df['movieId'].cat.categories[i] for i in movie_cat_codes]
  return movie_name, list(movie_meta_df.loc[movie_meta_df['id'].isin([str(id) for id in ids])].original_title)
 
def get_movie_names_from_movie_category_codes(movie_cat_codes):
  ids = [ratings_df['movieId'].cat.categories[i] for i in movie_cat_codes]
  return list(movie_meta_df.loc[movie_meta_df['id'].isin([str(id) for id in ids])].original_title)

def get_recommendation_for_user(user_id):
  user_cat_code = user_category_to_code[int(user_id)]
  movie_cat_codes = model.recommend(user_cat_code, r.tocsr().getrow(user_cat_code))[0]
  ids = [ratings_df['movieId'].cat.categories[i] for i in movie_cat_codes]
  return list(movie_meta_df.loc[movie_meta_df['id'].isin([str(id) for id in ids])].original_title)

def get_similar_movies_for_user(user_id, movie_name):
  search_result =search_movie(movie_name)
  if len(search_result) > 0:
    movie_id = search_result[0]['id']
    movie_name = search_result[0]['original_title']
  else:
    return []
  movie_cat_code = movie_category_to_code[int(movie_id)]
  user_cat_code = user_category_to_code[int(user_id)]
  movie_cat_codes = model.rank_items(user_cat_code, r.tocsr().getrow(user_cat_code), model.similar_items(movie_cat_code)[0])[0]
  ids = [ratings_df['movieId'].cat.categories[i] for i in movie_cat_codes]
  return list(movie_meta_df.loc[movie_meta_df['id'].isin([str(id) for id in ids])].original_title)

tab1, tab2, tab3 = st.tabs(["Similar Movies", "Recommend for User", "Get similar movies for User"])

with tab1:
    movie_name = st.selectbox('Movie title', list(movie_meta_df['original_title'].head(50)))
    st.write('Similar movies:', find_similar_movie(movie_name)[1])

with tab2:
    user_id = st.selectbox('Enter User Id', list(ratings_df['userId'].unique()))
    st.write('Recommendations for user', get_recommendation_for_user(user_id))

with tab3:
    user_id = st.selectbox('Enter user id', list(ratings_df['userId'].unique()))
    movie_name = st.selectbox('Enter a movie liked by the user', list(movie_meta_df['original_title'].head(50)))
    st.write(f'Movies similar to "{movie_name}" for user', get_similar_movies_for_user(user_id, movie_name))