import numpy

def cj_id(cj_ids, arg_id, arg_key=-1):
    result = []
    for id in cj_ids['uids']:
        if id['id'] == arg_id and id['key'] == arg_key:
            result += [id['value']]
    return result

spark.udf.register("cj_id", cj_id, ArrayType(StringType()))

def cj_attr(cj_attributes, arg_id, arg_key=None):
    result = []
    if cj_attributes is not None:
        for attr in cj_attributes:
            for member_id in range(0, 8):
                member_name = 'member' + str(member_id)
                if attr is not None and member_name in attr:
                    if attr[member_name] is not None and 'id' in attr[member_name]:
                        if attr[member_name]['id'] == arg_id and ('key' not in attr[member_name] or attr[member_name]['key'] == arg_key):
                            result += [attr[member_name]['value']]
    return result

spark.udf.register("cj_attr", cj_attr, ArrayType(StringType()))

def test(attributes, arg_id, type):
    if arg_id in [x['member'+type]['id'] for x in  attributes if x['member'+type] != None]:
        return "1"
    else:
        return "0"





# Check Nulls
# train_py.isnull().sum()


train_py['target'] = 1.0
scoring_py['target'] = 0.0

# Check distribution of 2 datasets

# Map Train Data
# train_py.active_card[~train_py.active_card.isin(['No','Yes'])] = "NA"
train_py.tl_account[train_py.tl_account=="No"] = 0.0
train_py.tl_account[train_py.tl_account=="Yes"] = 1.0
train_py.active_card[train_py.active_card=="Has problem"] = "problem"
# train_py.has_loans[train_py.has_loans=="No"] = 0
# train_py.has_loans[train_py.has_loans=="Yes"] = 1
# train_py['num_loans'] = train_py.num_loans - 1
# train_py.num_loans[train_py.num_loans < 0] = 0

# Map Scoring to train
# scoring_py['app_ts'] = date()
# scoring_py['has_loans'] = numpy.where(scoring_py.loans_num > 0, 1, 0)
# scoring_py.drop(columns=["loans_num"], inplace=True)
scoring_py.tl_account[scoring_py.tl_account=="no"] = 0.0
scoring_py.tl_account[scoring_py.tl_account=="ok"] = 1.0
scoring_py.active_card[scoring_py.active_card=="no"] = "No"
scoring_py.active_card[scoring_py.active_card=="ok"] = "Yes"
scoring_py.active_card[scoring_py.active_card=="return"] = "Returned"
# scoring_py.active_card[scoring_py.active_card=="no"] = 0
# scoring_py.active_card[scoring_py.active_card.isin(["ok","return"])] = 1

# Concatentate
df_py = pandas.concat([train_py, scoring_py])

# Data Quality
df_py.credit_score = pandas.to_numeric(df_py.credit_score)
df_py.card_limit = pandas.to_numeric(df_py.card_limit)
df_py.tl_account = pandas.to_numeric(df_py.tl_account)
# df_py['blank_score'] = numpy.where(df_py.credit_score <= 1, 1, 0)
df_py['blank_limit'] = numpy.where(df_py.card_limit.isnull(), 1, 0)
df_py.fillna({"card_limit":round(df_py.card_limit.mean(skipna=True), 2), "usable_card_limit":round(df_py.usable_card_limit.mean(skipna=True))}, inplace=True)
# df_py.credit_score[df_py.blank_score==1] = round(df_py.credit_score[df_py.blank_score == 0].mean())
# numpy.histogram(df_py.credit_score, bins=20)
# numpy.histogram(df_py.card_limit, bins=20)


df_py = pandas.get_dummies(df_py, columns=['active_card'])

df = df_py




numeric_attrs = list(df_py.select_dtypes(include=['uint8','float64', 'int64']).columns)
numeric_attrs.remove('target')
numeric_attrs.remove('crmid')











