# Script to Load Customer Journey Data

# Output: train_df, score_df, cj, cp

import datetime
import pandas
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import lit, to_date




# Input parameters
train_dir = "/analytics/"
train_features = "train.csv"
# train_target = "ids.csv"
scoring_dir = "/analytics/"
# scoring_f1 = "Onlystep4completion.csv"
# scoring_f2 = "allstepscompletion.csv"
scoring_features = "scoring.csv"

# Load Features Data
train_df = spark.read.option("delimiter",";").option("header","true").csv("/".join([train_dir, train_features]))
train_df = train_df.toDF("crmid","id","app_ts","source","tl_account","credit_score","has_savings_account","num_loans","active_card","card_limit","usable_card_limit")
train_df = train_df.drop("id","source","has_savings_account")
train_df = train_df.withColumn("app_ts",to_date(train_df["app_ts"]))
train_df = train_df.select("crmid","app_ts","tl_account","credit_score","num_loans","active_card","card_limit","usable_card_limit")

train_df.createOrReplaceTempView("train_df")

scoring_df = spark.read.option("delimiter",";").option("header","true").csv("/".join([scoring_dir, scoring_features]))
scoring_df = scoring_df.toDF("crmid", "id", "tl_account", "credit_score", "num_loans", "active_card", "card_limit","usable_card_limit")
scoring_df = scoring_df.withColumn("app_ts",to_date(lit("2019-02-13")))
scoring_df = scoring_df.select("crmid", "app_ts", "tl_account", "credit_score", "num_loans", "active_card", "card_limit", "usable_card_limit")
# scoring_df1 = spark.read.option("delimiter",";").option("header","true").csv("/".join([scoring_dir, scoring_f1]))
# scoring_df2 = spark.read.option("delimiter",";").option("header","true").csv("/".join([scoring_dir, scoring_f2]))



# Load Target Data
# target = spark.read.option("delimiter",";").option("header","true").csv("/".join([train_dir, train_target]))

cj_path = '/data/6287566b-8263-497b-8a4e-fbdb2680df1e/.dmpkit/customer-journey/master/cdm'
cp_path = '/data/6287566b-8263-497b-8a4e-fbdb2680df1e/.dmpkit/profiles/master/cdm/'

# Load Customer Journey
cj = spark.read.format("com.databricks.spark.avro").load(cj_path)
time_from = int(datetime.datetime(2018, 12, 25).timestamp()) * 1000
time_to = int(datetime.datetime(2018, 12, 30).timestamp()) * 1000
cj = cj.filter('ts > {} and ts < {}'.format(time_from, time_to))
cj.createOrReplaceTempView('cj')

# Load Customer Profile
cp_part = spark.read.format("com.databricks.spark.avro").load(cp_path)
time_from = int(datetime.datetime(2018, 12, 30).timestamp()) * 1000
time_to = int(datetime.datetime(2018, 12, 31).timestamp()) * 1000
cp = cp_part.filter('ts > {} and ts < {}'.format(time_from, time_to))
cp.createOrReplaceTempView('cj')




