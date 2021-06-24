import numpy as np

from scipy import sparse


def encode_column(column):
    
    keys=column.unique()
    keys_to_id={key:idx for idx ,key in enumerate(keys)}
    return keys_to_id, np.array([keys_to_id[x] for x in column]),len(keys)

def encode_df(anime_df):

    anime_ids,anime_df['anime_id'],num_anime=encode_column(anime_df['anime_id'])
    user_ids,anime_df['user_id'], num_users=encode_column(anime_df['user_id'])
    return anime_df,num_users,num_anime,user_ids,anime_ids

def create_embedding(n,k):
    return 11*np.random.random((n,k))/k

def create_sparse_matrix(df,rows,cols,column_name='rating'):
    return sparse.csc_matrix((df[column_name].values,(df['user_id'].values, df['anime_id'].values)),shape=(rows, cols))

def predict(df,emb_user,emb_anime):
    
    df['prediction'] = np.sum(np.multiply(emb_anime[df['anime_id']],emb_user[df['user_id']]), axis=1)
    return df

lmbda=0.0002

def cost(df,emb_user,emb_anime):
    Y=create_sparse_matrix(df,emb_user.shape[0],emb_anime.shape[0])
    predicted=create_sparse_matrix(predict(df,emb_user,emb_anime),emb_user.shape[0],emb_anime.shape[0],'prediction')
    return np.sum((Y-predicted).power(2))/df.shape[0]

def gradient(df, emb_user, emb_anime):
    """ Computes the gradient for user and anime embeddings"""
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_anime), emb_user.shape[0], emb_anime.shape[0], 'prediction')
    delta =(Y-predicted)
    grad_user = (-2/df.shape[0])*(delta*emb_anime) + 2*lmbda*emb_user
    grad_anime = (-2/df.shape[0])*(delta.T*emb_user) + 2*lmbda*emb_anime
    return grad_user, grad_anime



def gradient_descent(df, emb_user, emb_anime, iterations=2000, learning_rate=0.01, df_val=None):
    """ 
    Computes gradient descent with momentum (0.9) for given number of iterations.
    emb_user: the trained user embedding
    emb_anime: the trained anime embedding
    """
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    beta = 0.9
    grad_user, grad_anime = gradient(df, emb_user, emb_anime)
    v_user = grad_user
    v_anime = grad_anime
    for i in range(iterations):
        grad_user, grad_anime = gradient(df, emb_user, emb_anime)
        v_user = beta*v_user + (1-beta)*grad_user
        v_anime = beta*v_anime + (1-beta)*grad_anime
        emb_user = emb_user - learning_rate*v_user
        emb_anime = emb_anime - learning_rate*v_anime
        if(not (i+1)%50):
            print("\niteration", i+1, ":")
            print("train mse:",  cost(df, emb_user, emb_anime))
            if df_val is not None:
                print("validation mse:",  cost(df_val, emb_user, emb_anime))
    return emb_user, emb_anime


def encode_new_data(valid_df, user_ids, anime_ids):
    """ Encodes valid_df with the same encoding as train_df.
    """
    df_val_chosen = valid_df['anime_id'].isin(anime_ids.keys()) & valid_df['user_id'].isin(user_ids.keys())
    valid_df = valid_df[df_val_chosen]
    valid_df['anime_id'] =  np.array([anime_ids[x] for x in valid_df['anime_id']])
    valid_df['user_id'] = np.array([user_ids[x] for x in valid_df['user_id']])
    return valid_df

