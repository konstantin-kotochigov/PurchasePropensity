# Imports
import pandas
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensembles import RandomForestClassifier, GradientBoostingClassifer
from sklearn.metrics import roc_auc_score, precision_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, validation_curve

# Cross-correlation
df = df_py
X = df[numeric_attrs]
y = df.target


# Univariate Feature Importance
numeric_importance = pandas.DataFrame({"attr":numeric_attrs})
nominal_importnace = pandas.DataFrame({"attr":nominal_attrs})

lr =  LogisticRegression(solver='lbfgs')

rocauc_importance = []
for attr in numeric_attrs:
    print("Processing ",attr)
    lr.fit(X[attr].to_frame(), y)
    print(lr.intercept_)
    print(lr.coef_)
    y_pred = lr.predict_proba(df[attr].to_frame())[:,1]
    rocauc_importance.append(roc_auc_score(y, y_pred))

numeric_importance['auc'] = rocauc_importance

# Feature generation

# GridSearchCV




# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_resample(X, y)

# X = pandas.concat([X, X[y==1]], axis=0)
# y = pandas.concat([y, y[y==1]], axis=0)


y_pred = lr.predict_proba(X)[:,1]
prec, rec, tre = precision_recall_fscore_support(y, y_pred)
f_score = [2*x*y/(x+y) for (x,y) in zip(prec,rec)]


matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.clf()
plt.style.use('bmh')
plt.plot(tre, prec[:-1], 'b--', label='precision')
plt.savefig("plots/prec_curve.png", dpi=300)

plt.clf()
plt.plot(tre, rec[:-1], 'g--', label='recall')
plt.savefig("plots/prec_recall.png")

plt.clf()
plt.plot(tre, f_score[:-1], 'b--', label='f1-score')
plt.savefig("plots/f1_curve.png")

plt.clf()
plt.plot(tre, prec[:-1], 'b--', label='precision')
plt.plot(tre, rec[:-1], 'g--', label='recall')
plt.savefig("plots/prec_recall_curve.png")

# Plot scoring density
kde = KernelDensity(kernel="gaussian", bandwidth=0.15)
kde.fit(y_pred[:,np.newaxis])
X_plot = numpy.linspace(0, 1.0, 100)[:, np.newaxis]
d = numpy.exp(kde.score_samples(X_plot))
plt.clf()
plt.xlabel("Probability")
plt.ylabel("Density")
# plt.xticks([])
plt.plot(X_plot, d)
plt.hist(y_pred, bins=20)
plt.savefig("plots/prob_density.png", dpi=300)

# Evaluate

# Plot graphs