from sklearn.model_selection import GridSearchCV

def run_grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Runs GridSearchCV and returns the best estimator and score.

    Parameters:
    - model: sklearn model
    - param_grid: dict, hyperparameter grid
    - X_train: features
    - y_train: labels
    - cv: number of cross-validation folds
    - scoring: evaluation metric

    Returns:
    - best_estimator_: model with best found parameters
    - best_params_: dict of best hyperparameters
    - best_score_: float of best CV score
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
