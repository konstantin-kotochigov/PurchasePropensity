cj_df = spark.sql('''
select
    cj_id(id, 10008, 10032)[0] as crmid,
    cj_attr(attributes, 10035)[0] as device_type,
    date(from_unixtime(ts/1000)) as ts
from cj c
''').filter("crmid is not null")


cj_features_raw = spark.sql('''select cj_id(id, 10008, 10032)[0] as crmid, date(from_unixtime(ts/1000)) as ts from cj''').filter("crmid is not null and crmid!='undefined'")
cj_features_raw_py = cj_features_raw.collect()
cj_ids = pandas.DataFrame([(float(x[0]),x[1]) for x in cj_features_raw_py], columns=['crmid','ts'])

available_train_py = train_py[train_py.app_ts >= datetime.date(2018,12, 25)][["crmid","app_ts"]]

cj_df = cj_ids.join(available_train_py.set_index("crmid"), on="crmid", how='inner').filter("ts < app_ts")

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