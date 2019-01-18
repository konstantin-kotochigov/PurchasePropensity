# Imports
import pandas
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Parameters
input_dir = "./data"
input_features = "features.csv"
input_target = "target"

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

df['has_limit'] = 1
df['has_limit'][df.limit.isna()] = 0

df['has_usable_limit'] = 1
df['has_usable_limit'][df.usable_limit.isna()] = 0

# Classify column types
nominal_attrs = list(df.select_dtypes(include=['object','category']).columns)
numeric_attrs = list(df.select_dtypes(include=['float64']).columns)
numeric_attrs.remove('target')

dummy_df = pandas.get_dummies(data=df[nominal_attrs], drop_first=True)
dummy_df.columns = ["dummy_"+x for x in dummy_df.columns]
dummy_attrs = list(dummy_df.columns)
df = pandas.concat([df, dummy_df], axis=1)

numeric_attrs = numeric_attrs + dummy_attrs

# Cross-correlation


# Univariate Feature Importance
numeric_importance = pandas.DataFrame({"attr":numeric_attrs})
nominal_importnace = pandas.DataFrame({"attr":nominal_attrs})

lr =  LogisticRegression()

rocauc_importance = []
for attr in numeric_attrs:
    notnull_rows = df[attr].isna()==False
    lr.fit(df[attr][notnull_rows].to_frame(), df.target[notnull_rows])
    y_pred = lr.predict(df[attr][notnull_rows].to_frame())
    rocauc_importance.append(roc_auc_score(df.target[notnull_rows], y_pred))
numeric_importance['auc'] = rocauc_importance

# Feature generation

# GridSearchCV

# Evaluate

# Plot graphs