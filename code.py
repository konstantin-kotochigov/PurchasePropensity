# Imports
import pandas

# Input DF
df = pandas.read_csv("", sep="")
cj_features = pandas.read_csv("", sep="")
target = pandas.read_csv("")

df = features.merge(cj_features, by="id", how="left").merge(target, by="id", how="left")
df[df.target.isnull(), 'target'] = 0

# Classify column types


# Cross-correlation

# Univariate Feature Importance

# Feature generation

# GridSearchCV

# Evaluate

# Plot graphs