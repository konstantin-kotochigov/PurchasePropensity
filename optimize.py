import pandas
from sklearn.linear_model import ElasticNet, Lars, HuberRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn import metrics

def optimize_classifier(classifier, X, y, param_grid):
        lr_cv = GridSearchCV(classifier, param_grid, scoring=make_scorer(metrics.roc_auc_score), cv=5, verbose=2, n_jobs=2)
        model = lr_cv.fit(X, y)
        cv_results = lr_cv.cv_results_
        cv_table = pandas.DataFrame({"algo":classifier.__class__.__name__,"param":cv_results['params'], "quality":cv_results['mean_test_score'], "error_std":cv_results['std_test_score']}).sort_values(by="quality", ascending=False)
        return (cv_table)

# ----------------------------------------------------------------------------------------------------------------------

optimization_features = numeric_attrs

optimization_X = df[optimization_features]
optimization_y = df.target

# Define three parameter Grids
param_grid = dict()
param_grid['logistic'] = {"penalty":['l1','l2'], "C":[0.01, 0.05, 0.1, 0.2]}
param_grid['svc'] = {"kernel":["linear","rbf","poly"],"C":[0.01, 0.05, 0.1, 0.2]}
param_grid['elastic'] = {"alpha":[0.0, 0.5, 1.0], "l1_ratio":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],}
param_grid['randomforest'] = {'max_depth':(2,4,6,8,10), 'n_estimators':[100,500]}
param_grid['boosting'] = {'max_depth':(4,8,12), 'n_estimators':[100,500]}

# Models
models = dict()
models['logistic'] = LogisticRegression(max_iter=200, solver='liblinear')
models['svc'] = SVC()
models['elastic'] = ElasticNet()
models['randomforest'] = RandomForestClassifier(n_jobs=2)
models['boosting'] = GradientBoostingClassifier()

# Optimize Models
models_to_fit = ['logistic']
cv_table = pandas.DataFrame()
for model in models_to_fit:
    cv_table = cv_table.append(optimize_classifier(param_grid=param_grid[model], classifier=models[model], X=optimization_X, y=optimization_y))
    cv_table.sort_values(by="quality").to_csv("/home/deployer/cv_table_"+model+".csv", sep=";", index=False)





