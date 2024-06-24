import numpy as np


class CustomLogReg:
    def __init__(self, lambda_l2:float, abs_logit=10) -> None:
        self.lambda_l2 = lambda_l2

        self.act = lambda x: 1 / (1 + np.exp(-x))

        self.abs_logit = abs_logit

    def fit(self, X:np.array, Y:np.array) -> None:
        Y_logit = Y.copy()
        Y_logit[Y == 0] = -self.abs_logit
        Y_logit[Y == 1] = self.abs_logit
        Y_logit[Y == 0.5] = 0
        
        X_b = np.concatenate([X.copy(), np.ones((X.shape[0], 1))], axis=1)
        self.W = np.linalg.inv(X_b.T @ X_b + self.lambda_l2 * np.identity(X_b.shape[1])) @ X_b.T @ Y_logit

    def predict_proba(self, X:np.array) -> np.array:
        
        X_b = np.concatenate([X.copy(), np.ones((X.shape[0], 1))], axis=1)
        
        return self.act(X_b @ self.W)

    def predict(self, X:np.array, threshold=0.5) -> np.array:

        return (self.predict_proba(X) > threshold).astype(float)
    
    def auth(self, X:np.array, secret_key:np.array, p_threshold=0.5, part_right_threshold=0.9) -> np.array:
        
        proba = self.predict_proba(X)
        pred = (proba > p_threshold).astype(np.float16)
        secret_key = np.concatenate([secret_key] * proba.shape[0], axis=0)
        part_right = (pred == secret_key).sum(axis=1) / proba.shape[1]

        return (part_right >= part_right_threshold).astype(np.float16)


def fit_predict(
        X_train, Y_train,
        X_valid,
        X_test,
        lambda_l2=1e-8,
    ):

    log_reg = CustomLogReg(lambda_l2=lambda_l2, abs_logit=1)
    log_reg.fit(X_train, Y_train)

    Y_pred_train = log_reg.predict_proba(X_train)
    Y_pred = log_reg.predict_proba(X_valid)
    Y_pred_test = log_reg.predict_proba(X_test)

    return Y_pred_train, Y_pred, Y_pred_test


def p_errors(Y_true, Y_pred, X, person_id, secret_key, p_threshold=0.5, part_right_threshold=1, eps=1e-5):


     # делаем матрицу с одним и тем же ключом везде, чтобы удобно сравнивать
     secret_key = np.concatenate([secret_key] * Y_true.shape[0], axis=0)

     pred_key = (Y_pred >= p_threshold).astype(np.float16)
     
     # доля верных ответов в каждой строке
     part_right = (secret_key == pred_key).sum(axis=1) / Y_true.shape[1]

     pos = part_right >= part_right_threshold
     neg = 1 - pos

     FP = (pos & (X['name'] != person_id)).sum()
     FN = (neg & (X['name'] == person_id)).sum()

     p_i_err = FN / neg.sum()
     p_ii_err = FP / pos.sum()

     return p_i_err, p_ii_err