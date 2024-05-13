from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import pickle5 as pickle


class MF:

    def __init__(self, path):
        self.path = path
        self.user_id_map = None
        self.item_id_map = None

    def preprocess_data(self, path=None):
        path = self.path if path is None else path
        data = pd.read_csv(path)

        self.user_id_map = {user_id: index for index, user_id in enumerate(data['user_id'].unique())}
        self.item_id_map = {item_id: index for index, item_id in enumerate(data['item_id'].unique())}

        data['user_index'] = data['user_id'].map(self.user_id_map)
        data['item_index'] = data['item_id'].map(self.item_id_map)

        sparse_matrix = csr_matrix((data['rating'], (data['user_index'], data['item_index'])),
                                   shape=(len(self.user_id_map), len(self.item_id_map)))
        return sparse_matrix

    def train(self, factors=100, regularization=0.01):
        # ALS model
        sparse_matrix = self.preprocess_data()
        model = AlternatingLeastSquares(factors=factors, regularization=regularization)
        model.fit(sparse_matrix)
        return model

    def get_embeddings_map(self, id_map, factors):
        embeddings_dict = {}

        for id, index in id_map.items():
            embedding = factors[index]
            embeddings_dict[id] = embedding

        return embeddings_dict

    def get_latent_factors(self, model):
        user_embeddings_dict, item_embeddings_dict = self.get_embeddings_map(self.user_id_map, model.user_factors), self.get_embeddings_map(self.item_id_map,
                                                                                                      model.item_factors)
        self.save_embeddings(user_embeddings_dict,
                           config['save_path'] + '/' + config['source_domain'] + '_als_user_embeddings.pkl')
        self.save_embeddings(item_embeddings_dict,
                           config['save_path'] + '/' + config['source_domain'] + '_als_item_embeddings.pkl')

    def save_embeddings(self, embeddings_dict, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(embeddings_dict, file)

    def load_embeddings(self, file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    def calculate_rmse(self, model, path=None):
        true_matrix = self.preprocess_data(path)
        user_factors = model.user_factors
        item_factors = model.item_factors
        predicted_ratings = np.dot(user_factors, item_factors.T)

        user_indices, item_indices = true_matrix.toarray().nonzero()
        predicted_ratings_subset = predicted_ratings[user_indices, item_indices]
        actual_ratings_subset = true_matrix.toarray()[user_indices, item_indices]

        rmse = np.sqrt(np.mean((predicted_ratings_subset - actual_ratings_subset) ** 2))
        return rmse
