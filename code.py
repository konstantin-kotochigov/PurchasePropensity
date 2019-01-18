# Imports
import pandas
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

# Parameters
input_dir = "./data"
input_features = "features.csv"
input_target = "target.csv"

# Load Data
features = pandas.read_csv("/".join([input_dir, input_features]), sep=";")
# cj_features = pandas.read_csv("", sep="")
target = pandas.read_csv("/".join([input_dir, input_target]), sep=";")["dummy id"].rename("id").to_frame()
target['target'] = 1

features.columns = ['id','account','credit_score','loans','card','limit','usable_limit']

df = features.merge(target, on="id", how="left")
df.target[df.target.isna()] = 0

df.target.value_counts()

def histogram(x):
    hist = x.value_counts().rename("cnt")
    hist_n = round(100 * x.value_counts(normalize=True), 5).astype(str).rename("pct")
    df = pandas.concat([hist, hist_n+"%"], axis=1)[0:10]
    df['value'] = df.index
    df = df.reset_index()
    return(df[['value','cnt','pct']])

# For presentation
histogram(features.loans)
histogram(features.account)
histogram(features.card)

df['loans'] = pandas.cut(df.loans, [-1,0,1,1000], labels=['loans=0','loans=1','loans=2+'])

# df = features.merge(cj_features, by="id", how="left").merge(target, by="id", how="left")
# df[df.target.isnull(), 'target'] = 0

# Classify column types
nominal_attrs = list(df.select_dtypes(include=['object','category']).columns)
numeric_attrs = list(df.select_dtypes(include=['float64']).columns)
numeric_attrs.remove('target')

dummy_df = pandas.get_dummies(data=df[nominal_attrs], drop_first=True)
dummy_df.columns = ["dummy_"+x for x in dummy_df.columns]
dummy_attrs = list(dummy_df.columns)
df = pandas.concat([df, dummy_df], axis=1)

numeric_attrs = numeric_attrs + dummy_attrs

# Mark missing data
df['has_limit'] = 1
df['has_limit'][df.limit.isna()] = 0
df['has_usable_limit'] = 1
df['has_usable_limit'][df.usable_limit.isna()] = 0

# Fill missing data
df.fillna(df[numeric_attrs].mean(), inplace=True)
# df.fillna(df[nominal_attrs].mode(), inplace=True)


# Cross-correlation


# Univariate Feature Importance
numeric_importance = pandas.DataFrame({"attr":numeric_attrs})
nominal_importnace = pandas.DataFrame({"attr":nominal_attrs})

lr =  LogisticRegression(solver='lbfgs')

rocauc_importance = []
for attr in numeric_attrs:
    print("Processing ",attr)
    lr.fit(df[attr].to_frame(), df.target)
    y_pred = lr.predict_proba(df[attr].to_frame())[:,1]
    rocauc_importance.append(roc_auc_score(df.target, y_pred))

numeric_importance['auc'] = rocauc_importance

# Feature generation

# GridSearchCV

X = df[numeric_attrs]
y = df.target

# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_resample(X, y)

X = pandas.concat([X, X[y==1]], axis=0)
y = pandas.concat([y, y[y==1]], axis=0)

lrParamGrid={"penalty":("l1","l2"), "C":(0.1,0.5,1.0)}
lrCV = GridSearchCV(lr, param_grid=lrParamGrid, scoring='roc_auc', cv=5, verbose=0, n_jobs=-1)
lrCV.fit(X,y)
cv_results = lrCV.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
# cv_table.to_csv("re/gridsearch/coordinates_randomforest.csv", index=False)


# Evaluate

# Plot graphs