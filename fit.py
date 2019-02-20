import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

model_features = optimization_features.copy()
# model_features.remove("ts")

X = df[model_features].copy()
y = df.target

# Since GridSearch does not give normal results
manual_cv_results = []
for cv_num in range(15):
    print("cv_num=",cv_num)
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LogisticRegression(C=0.1, solver='lbfgs', n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    current_result = roc_auc_score(y_test, y_pred)
    print(current_result)
    manual_cv_results.append(current_result)

print(round(numpy.array(manual_cv_results).mean(), 5), round(numpy.array(manual_cv_results).std(),5))


# Fit models with Optimal Parameters
models_to_fit = ['logistic']
models = dict()
models['logistic']      =  LogisticRegression(C=0.1, solver='lbfgs')
models['randomforest']  =  RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1)
models['boosting']      =  GradientBoostingClassifier(max_depth=5, n_estimators=100)

for model in models_to_fit:
    print("Fitting ",model)
    models[model].fit(X[model_features],y)
    X[model+'_pred'] = models[model].predict(X[model_features])
    print(roc_auc_score(y, X[model + "_pred"]))

model = models['logistic']

scoring_py = df[df.target==0]
df['y_pred'] = model.predict_proba(df[model_features])[:,1]
scoring_py['y_pred'] = model.predict_proba(scoring_py[model_features])[:,1]

result = scoring_py.sort_values(by="y_pred", ascending=False)[['crmid', 'y_pred']]






df = df.sort_values(by="y_pred", ascending=False)
result.iloc[0]
result.iloc[-1]
df.loc[742819]
df.loc[658012]

result = scoring_py.sort_values(by="y_pred", ascending=False)[['crmid']][0:200000]

result_df = spark.createDataFrame(result)

# HADOOP_USER_NAME=hdfs
# result_df.repartition(10).write.parquet("/honme/deployer/case5_segment_crmids")



prec, rec, tre = precision_recall_curve(df.target, df.y_pred)
f_score = [2*x*y/(x+y) for (x,y) in zip(prec,rec)]

f_max = numpy.array(f_score).argmax()
prec[f_max]
rec[f_max]
tre[f_max]
df[df.y_pred > tre[f_max]].shape[0]

# Get Top-100000
top_n = 4425
tre_index = numpy.argmax(tre > df.iloc[top_n].y_pred)
prec[tre_index]
rec[tre_index]
f_score[tre_index]

# Print Coordinates
for x in tre[len(tre)-100000:len(tre):1000]:
    print(x)
for x in prec[len(tre)-100000:len(tre):1000]:
    print(x)
for x in rec[len(tre)-100000:len(tre):1000]:
    print(x)
for x in [sum(df.y_pred > x) for x in tre[len(tre)-100000:len(tre):1000]]:
    print(x)

def get_quliaty(df, top_n):
    current_df = df.copy()
    tre = df.iloc[top_n].y_pred
    current_df['y_pred_class'] = -1
    current_df.y_pred_class[current_df.y_pred > tre] = 1
    current_df.y_pred_class[current_df.y_pred <= tre] = 0

files = {}
files[0] = open("/home/deployer/result200000","w")
# files[1] = open("/home/deployer/result20000","w")
# files[2] = open("/home/deployer/result5000","w")

result = result.reset_index()
result = result.drop(columns="index")

for i,x in result.iterrows():
    if i > 200000:
        break
#     if i % 1000==0:
#         print(i)
#     if i < 15000:
#         ind = i % 3
#     elif i < 45000:
#         ind = i % 2
#     elif i < 120000:
#         ind = 0
    # if (14500 < i < 15000) or (35000 < i < 35500) or (75000 < i < 75500):
    #     print(ind)
    dummy = files[0].write(str(i) + "," + str(int(x.crmid)) + "\n")

files[0].close()

for x in range(len(files)):
    files[x].close()