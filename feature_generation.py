import numpy
import math

# Check Nulls
# train_py.isnull().sum()

# Convert to Python dataframes
train_py = pandas.DataFrame(train_df.collect(), columns=train_df.columns, dtype='float64')
scoring_py = pandas.DataFrame(scoring_df.collect(), columns=scoring_df.columns, dtype='float64')
scoring_py = scoring_py.drop_duplicates(subset=["crmid"], keep=False)

# Remove Training customers from scoring dataset
scoring_ids = pandas.DataFrame(list(set(scoring_py.crmid).difference(set(train_py.crmid))), columns=['crmid'])
scoring_py = scoring_py.join(scoring_ids.set_index("crmid"), how='inner', on='crmid')
scoring_py.reset_index()


train_py['target'] = 1.0
scoring_py['target'] = 0.0

# Check distribution of 2 datasets

# Map Train Data
# train_py.active_card[~train_py.active_card.isin(['No','Yes'])] = "NA"
train_py.tl_account[train_py.tl_account=="No"] = 0.0
train_py.tl_account[train_py.tl_account=="Yes"] = 1.0
train_py.active_card[train_py.active_card=="Has problem"] = "problem"

# Map Scoring to train
scoring_py.tl_account[scoring_py.tl_account=="no"] = 0.0
scoring_py.tl_account[scoring_py.tl_account=="ok"] = 1.0
scoring_py.active_card[scoring_py.active_card=="no"] = "No"
scoring_py.active_card[scoring_py.active_card=="ok"] = "Yes"
scoring_py.active_card[scoring_py.active_card=="return"] = "Returned"
# scoring_py.active_card[scoring_py.active_card=="no"] = 0
# scoring_py.active_card[scoring_py.active_card.isin(["ok","return"])] = 1

# Concatentate
df = pandas.concat([train_py, scoring_py])

# Data Quality
df.credit_score = pandas.to_numeric(df.credit_score)
df.card_limit = pandas.to_numeric(df.card_limit)
df.tl_account = pandas.to_numeric(df.tl_account)
# df_py['blank_score'] = numpy.where(df_py.credit_score <= 1, 1, 0)
# df_py['blank_limit'] = numpy.where(df_py.card_limit.isnull(), 1, 0)
df.fillna({"card_limit":round(df.card_limit.mean(skipna=True), 2), "usable_card_limit":round(df.usable_card_limit.mean(skipna=True))}, inplace=True)
df['credit_score'] = numpy.where(df.credit_score < 1, df.credit_score[df.credit_score > 1].mean(), df.credit_score)
df['num_loans'] = df.num_loans.apply(math.sqrt)

df = pandas.get_dummies(df, columns=['active_card'])

numeric_attrs = list(df.select_dtypes(include=['uint8','float64', 'int64']).columns)
numeric_attrs.remove('target')
numeric_attrs.remove('crmid')











