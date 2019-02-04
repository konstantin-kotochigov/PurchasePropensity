# Script to Load Customer Journey Data

import datetime
import pandas
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import lit

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

spark.udf.register("cj_test", test, StringType())
t = spark.sql("select cj_test(attributes, 50005, '4') as cnt from cj_part")
t.filter(cnt > 0).count()

path = '/data/6287566b-8263-497b-8a4e-fbdb2680df1e/.dmpkit/customer-journey/master/cdm'
cj_part = spark.read.format("com.databricks.spark.avro").load(path)
time_from = int(datetime.datetime(2018, 12, 30).timestamp()) * 1000
time_to = int(datetime.datetime(2018, 12, 31).timestamp()) * 1000

cj_part = cj_part.filter('ts > {} and ts < {}'.format(time_from, time_to))
cj_part.createOrReplaceTempView('cj_part')

# Parameters
input_dir = "/data/6287566b-8263-497b-8a4e-fbdb2680df1e/Machine Learning_People who gets loan succesfully recently"
input_features = "DUMMY ID & Attrıbutes_dmpuploadfile.csv"
input_target = "ids.csv"

# Load Data
train_df = spark.read.option("delimiter",";").option("header","true").csv("/".join([input_dir, input_features]))
train_df = train_df.toDF("id","dt","source","savings","score","loans_flag","card_flag","card_limit")
train_df = train_df.drop("source")
train_df = train_df.withColumn("target", lit(1))

train_py = pandas.DataFrame(train_df.collect(), columns=train_df.columns)
train_py.show()
train_py

train_df.createOrReplaceTempView("train_df")

# Load target
target = spark.read.option("delimiter",";").option("header","true").csv("/".join([input_dir, input_target]))


left join train_df t on c.id=t.id
from_unixtime(ts/1000) as ts,

base = spark.sql('''
select
    id.gid as oper_b_globuserid,
    cj_id(id, 10008)[0] as f1,
    cj_attr(attributes, 10045)[0] as f2,
    cj_attr(attributes, 10052)[0] as f3,
    cj_attr(attributes, 10055)[0] as f4    
    ts as app_ts
from cj_part c
''')

base.fillna(app_ts => current_dt)
base = base.filter(app_ts > ts and ts > app_ts - 2W)

features = spark.sql("""
    select
        id,
        sum(f1) as f1_sum,
        sum(f2) as f2_sum.
        sum(f3) as f3_sum,
        sum(f4) as f4_sum,
        sum(f5) as f5_sum,
        max(f1) as f1_prev,
        max(f2) as f2_prev,
        max(f3) as f3_prev,
        max(f4) as f4_prev,
        max(f5) as f5_prev,
        count(*) as cnt
    from
        base
""")

features.coalesce(1).write.parquet("")

