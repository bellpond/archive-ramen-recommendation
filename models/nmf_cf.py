from sklearn.decomposition import NMF
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NMFBaseCF():
    def __init__(self, data, latent_dim: int=16, normalize=False, binarize=False, random_state=0, verbose=20):
        '''
        Args:
            data: pandas.DataFrame
                Its columns must contain 'user_id', 'store_id' and 'score'.
            latent_dim: int
                The num of latent space dimension of NMF. It should be smaller than the num of unique "user_id" and unique "store_id".
            normalize: boolean
                If True, normalize the scores.
            binarize: boolean
                If True, binarize the score.
            random_state: int
                Random seed
        '''

        # check
        assert 'user_id' in data.columns, 'The "user_id" column is not included in the input DataFrame.'
        assert 'store_id' in data.columns, 'The "store_id" column is not included in the input DataFrame.'
        assert 'score' in data.columns, 'The "score" column is not included in the input DataFrame.'

        if normalize:
            print("Min-Max Normalizing the score data.")
            data = self._normalize_score(data)

        if binarize:
            print("Binarize the score data.")
            data = self._binarize_score(data)

        self.latent_dim = latent_dim
        self.model = NMF(n_components=latent_dim, random_state=random_state, verbose=verbose)

        self.ratings = pd.pivot_table(data, index='user_id', columns='store_id', values='score', fill_value=0)
        self.user_id = self.ratings.index
        self.store_id = self.ratings.columns

        self.pred = None

    def fit(self):
        W = self.model.fit_transform(self.ratings)
        H = self.model.components_
        self.pred = pd.DataFrame(np.dot(W, H), index=self.user_id, columns=self.store_id)

    def predict_score(self, user_id, top_n: int=None, include_known=False):
        assert self.pred is not None, "Fit the model before making prediction."
        score = self.pred.loc[user_id,:]
        
        if include_known:
            mask = np.full(len(score), True)
        else:
            mask = self.ratings[user_id] == 0  # unscored store idx

        if top_n is None:  # return all data
            return score[mask]
        else:  # return top n data
            return score[mask].sort_values(ascending=False).head(top_n)

    def _normalize_score(self, df):
        scores = df['score']
        score_max = scores.max()
        score_min = scores.min()
        df['score'] = (scores - score_min) / (score_max - score_min)
        return df

    def _binarize_score(self, df):
        df['score'].loc[df['score'] > 0] = 1.0
        return df


    # def _make_test_case(self, n_test):
    #     # non-zero
    #     non_zero_indices = np.nonzero(self.ratings.values)
    #     random_indices = np.random.choice(range(len(non_zero_indices[0])), size=n_test, replace=False)
    #     test_case_indices = non_zero_indices[0][random_indices], non_zero_indices[1][random_indices]

    #     test_matrix = deepcopy(self.ratings.values)
    #     test_matrix[test_case_indices] = 0

    #     return test_matrix, test_case_indices

    # def test(self, n_test=1000, graph_path='./output/test.png'):
    #     rating_matrix_test, test_case_indices = self._make_test_case(n_test)

    #     W = self.model.fit_transform(rating_matrix_test)
    #     H = self.model.components_
    #     self.rating_matrix_pred = np.dot(W, H)

    #     pred = self.rating_matrix_pred[test_case_indices]
    #     gt = self.rating_matrix_org[test_case_indices]
    #     print(pred)
    #     print(gt)
    #     mse_loss = np.mean((pred - gt)**2)

    #     # correlation coefficient
    #     s1 = pd.Series(pred)
    #     s2 = pd.Series(gt)
    #     r = s1.corr(s2)

    #     self._make_scatter(pred, gt, graph_path)

    #     return mse_loss, r

