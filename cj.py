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
t = spark.sql("select cj_attr(attributes, 10008, 10032) as cnt from cj_part")
t.filter(cnt > 0).count()

cj_path = '/data/6287566b-8263-497b-8a4e-fbdb2680df1e/.dmpkit/customer-journey/master/cdm'
cp_path = '/data/6287566b-8263-497b-8a4e-fbdb2680df1e/.dmpkit/customer-journey/master/cdm'

# Load Customer Journey
cj = spark.read.format("com.databricks.spark.avro").load(cj_path)
time_from = int(datetime.datetime(2018, 12, 30).timestamp()) * 1000
time_to = int(datetime.datetime(2018, 12, 31).timestamp()) * 1000
cj = cj_part.filter('ts > {} and ts < {}'.format(time_from, time_to))
cj.createOrReplaceTempView('cj_part')

# Load Customer Profile
cj_part = spark.read.format("com.databricks.spark.avro").load(cp_path)
time_from = int(datetime.datetime(2018, 12, 30).timestamp()) * 1000
time_to = int(datetime.datetime(2018, 12, 31).timestamp()) * 1000
cp = cj_part.filter('ts > {} and ts < {}'.format(time_from, time_to))
cp.createOrReplaceTempView('cj_part')

# Parameters
train_dir = "/data/6287566b-8263-497b-8a4e-fbdb2680df1e/Machine Learning_People who gets loan succesfully recently"
train_features = "DUMMY ID & Attrıbutes_dmpuploadfile.csv"
train_target = "ids.csv"
scoring_dir = "/data/6287566b-8263-497b-8a4e-fbdb2680df1e/Macjine Learning"
scoring_features = "/data/6287566b-8263-497b-8a4e-fbdb2680df1e/Machine Learning_People who gets loan succesfully recently"

# Load Features Data
train_df = spark.read.option("delimiter",";").option("header","true").csv("/".join([input_dir, input_features]))
train_df = train_df.toDF("crmid","app_ts","source","savings","score","loans_flag","card_flag","card_limit")
train_df = train_df.drop("source")
train_df = train_df.withColumn("target", lit(1))

train_py = pandas.DataFrame(train_df.collect(), columns=train_df.columns)
train_py.show()
train_py

train_df.createOrReplaceTempView("train_df")

scoring_df = spark.read.option("delimiter",";").option("header","true").csv("/".join([input_dir, input_features]))

# Load Target Data
target = spark.read.option("delimiter",";").option("header","true").csv("/".join([input_dir, input_target]))


left join train_df t on c.id=t.id
from_unixtime(ts/1000) as ts,

cj_df = spark.sql('''
select
    cj_id(id, 10008, 10032)[0] as crmid,
    cj_attr(attributes, 10035)[0] as device_type,
    date(from_unixtime(ts/1000)) as ts
from cj_part c
''').filter("crmid is not null")

cj_df = cj_df.join(train_df.select("crmid","app_ts"), "crmid", how='inner').filter("ts < app_ts")

cj_df.createOrReplaceTempView("cj_df")

base.fillna(app_ts => current_dt)
base = base.filter(app_ts > ts and ts > app_ts - 2W)

features = spark.sql("""
    select
        crmid,
        app_ts,
        sum(case when device_type==10000 then 1 else 0 end) as f1_cnt,
        sum(case when device_type==10001 then 1 else 0 end) as f2_cnt,
        sum(case when device_type==10002 then 1 else 0 end) as f3_cnt,
        sum(case when device_type==10003 then 1 else 0 end) as f4_cnt,
        max(case when device_type==10000 then ts else to_date('1900-01-01') end) as f1_prev,
        max(case when device_type==10001 then ts else to_date('1900-01-01') end) as f2_prev,
        max(case when device_type==10002 then ts else to_date('1900-01-01') end) as f3_prev,
        max(case when device_type==10003 then ts else to_date('1900-01-01') end) as f4_prev,
        count(*) as cnt
    from
        cj_df
    group by
        crmid,
        app_ts
""")

features.coalesce(1).write.parquet("")

