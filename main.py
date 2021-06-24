import numpy as np

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

from scipy import sparse

from train import encode_df

from train import create_sparse_matrix

from train import create_embedding

from train import gradient_descent

from train import encode_new_data

from train import cost

anime_rating_df=pd.read_csv("/Users/huangbowei/Desktop/coding/Python/Matrix Factorization /data/rating.csv")



anime_rating = anime_rating_df.loc[anime_rating_df.rating != -1].reset_index()[['user_id','anime_id','rating']]



Counter(anime_rating.groupby(['user_id']).count()['anime_id'])

np.mean(anime_rating.groupby(['user_id']).count()['anime_id'])


train_df,valid_df=train_test_split(anime_rating,test_size=0.2)

train_df = train_df.reset_index()[['user_id', 'anime_id', 'rating']]
valid_df = valid_df.reset_index()[['user_id','anime_id','rating']]

train_df.head()
valid_df.head()

anime_df, num_users, num_anime, user_ids, anime_ids = encode_df(train_df)



anime_df, num_users, num_anime, user_ids, anime_ids = encode_df(train_df)

Y=create_sparse_matrix(anime_df,num_users,num_anime)

Y.todense()

emb_user=create_embedding(num_users,3)
emb_anime=create_embedding(num_anime,3)

emb_user,emb_anime=gradient_descent(anime_df,emb_user,emb_anime,iterations=800,learning_rate=1)


print('before:', valid_df.shape)
valid_df=encode_new_data(valid_df,user_ids,anime_ids)
print('after:', valid_df.shape)

train_mse = cost(train_df, emb_user, emb_anime)
val_mse = cost(valid_df, emb_user, emb_anime)
print(train_mse, val_mse)


print(valid_df[70:80].head())

