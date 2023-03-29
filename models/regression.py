import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)


def train_models(train_data, kind = 'updrs_1'):
    X = train_data.copy()
    y = X.pop(kind)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    param_grid = {
    'n_estimators': [5, 10, 50, 100],
    'max_depth': [2, 3, 4]
    }
    rf = RandomForestRegressor()
    smape_scorer = make_scorer(smape, greater_is_better=False)
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                               cv = 3, n_jobs = -1, verbose = 1, scoring = smape_scorer)
    grid_search.fit(X_train, y_train)
    model_ = grid_search.best_estimator_
    best_score = grid_search.best_score_
    
    return model_, best_score, scaler


def run_models():
    train_data = pd.read_csv("../data/artificial_data/train_merged.csv")
    train_updrs_1 = train_data.drop(['updrs_2','updrs_3','updrs_4','upd23b_medication_Off','upd23b_medication_On'], axis = 1)
    train_updrs_2 = train_data.drop(['updrs_1','updrs_3','updrs_4','upd23b_medication_Off','upd23b_medication_On'], axis = 1)
    train_updrs_3 = train_data.drop(['updrs_1','updrs_2','updrs_4','upd23b_medication_Off','upd23b_medication_On'], axis = 1)
    train_updrs_4 = train_data.drop(['updrs_1','updrs_2','updrs_3','upd23b_medication_Off','upd23b_medication_On'], axis = 1)
    model_uprds_1, best_score_uprds_1, scaler_uprds_1 = train_models(train_updrs_1, kind='updrs_1')
    model_uprds_2, best_score_uprds_2, scaler_uprds_2 = train_models(train_updrs_2, kind='updrs_2')
    model_uprds_3, best_score_uprds_3, scaler_uprds_3 = train_models(train_updrs_3, kind='updrs_3')
    model_uprds_4, best_score_uprds_4, scaler_uprds_4 = train_models(train_updrs_4, kind='updrs_4')

    print(best_score_uprds_1)
    print(best_score_uprds_2)
    print(best_score_uprds_3)
    print(best_score_uprds_4)


if __name__ == "__main__":
    run_models()

