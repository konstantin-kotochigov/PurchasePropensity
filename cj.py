# Script to Load Customer Journey Data

import datetime
from pyspark.sql.types import ArrayType, StringType

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

path = '/user/admin/data/Svaznoj/'
df = spark.read.format("com.databricks.spark.avro").load(path)
time_from = int(datetime.datetime(2018, 7, 5).timestamp()) * 1000
time_to = int(datetime.datetime(2018, 7, 6).timestamp()) * 1000

cj_part = df.filter('ts > {} and ts < {}'.format(time_from, time_to))
cj_part.createOrReplaceTempView('cj_part')

base = spark.sql('''
select
    id.gid as oper_b_globuserid,
    cj_attr(attributes, 10059)[0] as f1,
    cj_attr(attributes, 10060)[0] as f2,
    cj_attr(attributes, 10061)[0] as f3,
    cj_attr(attributes, 10062)[0] as f4,
    cj_attr(attributes, 10063)[0] as f5,
    from_unixtime(ts/1000) as ts,
    app.ts as app_ts
from cj_part
left join app on cj_part.od=app.id
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

