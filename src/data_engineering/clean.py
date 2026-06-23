# src/data_engineering/clean.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, from_unixtime

def clean_taobao_data(spark: SparkSession, raw_data_path: str) -> "DataFrame":
    """
    针对淘宝用户行为大数据的并行清洗函数
    
    参数:
        spark: 已经初始化的 SparkSession 实例
        raw_data_path: 原始数据的存储路径（如 data/raw/UserBehavior.csv）
    返回:
        pyspark.sql.DataFrame: 清洗完成后的标准 DataFrame
    """
    print(f"【数据清洗】正在从 {raw_data_path} 读取原始数据...")
    
    # 1. 读取海量 CSV 数据，并定义 Schema（淘宝数据集标准格式）
    # 包含：用户ID, 商品ID, 商品类目ID, 行为类型(pv/buy/cart/fav), 时间戳
    schema = "user_id LONG, item_id LONG, category_id LONG, behavior_type STRING, timestamp LONG"
    
    raw_df = spark.read \
        .format("csv") \
        .option("header", "false") \
        .schema(schema) \
        .load(raw_data_path)
    
    # 2. 基础数据清洗
    # 去除完全重复的行，并过滤掉关键字段含空值的脏数据
    cleaned_df = raw_df.dropDuplicates().dropna(subset=["user_id", "item_id", "timestamp"])
    
    # 3. 时间字段转换（将 Unix 时间戳转换为可读的日期格式，方便后续按时间窗口切分）
    # 增加：date_str (String 类型日期) 和 date (Date 类型)
    cleaned_df = cleaned_df \
        .withColumn("date_str", from_unixtime(col("timestamp"), "yyyy-MM-dd")) \
        .withColumn("date", to_date(col("date_str")))
    
    # 4. 根据实际业务场景剔除明显的时间异常值
    # 假设数据集的标准时间段在 2017-11-25 至 2017-12-03 之间，超出这个范围的属于系统脏数据
    cleaned_df = cleaned_df.filter(
        (col("date") >= "2017-11-25") & (col("date") <= "2017-12-03")
    )
    
    print(f"【数据清洗】清洗完成！已过滤重复值、空值及时间异常数据。")
    return cleaned_df

if __name__ == "__main__":
    # 这里的代码仅用于你在本地单独运行、测试 clean.py 脚本时生效
    # 当被别的脚本 import 调用时不会被执行
    print("【本地测试】正在初始化本地 SparkSession...")
    
    # 初始化本地测试用的 Spark 环境
    spark_test = SparkSession.builder \
        .master("local[*]") \
        .appName("LocalTestClean") \
        .getOrCreate()
        
    # 模拟运行（确保你已经把数据放到了 data/raw/ 目录下）
    try:
        test_df = clean_taobao_data(spark_test, "data/raw/UserBehavior.csv")
        test_df.show(5)  # 打印前 5 行看看结果
    except Exception as e:
        print(f"【测试失败】请检查数据路径是否正确。错误信息: {e}")
    finally:
        spark_test.stop()