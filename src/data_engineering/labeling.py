# src/data_engineering/labeling.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, when, lit

def generate_user_item_labels(cleaned_df, feature_end_date: str, label_end_date: str) -> "DataFrame":
    """
    基于滑动窗口对用户-商品对进行未来7天购买行为的打标
    
    参数:
        cleaned_df: 上一步 clean.py 输出的清洗后的 Spark DataFrame
        feature_end_date: 特征观测期的结束日期（格式 'yyyy-MM-dd'），例如 '2017-11-30'
        label_end_date: 标签预测期的结束日期（通常为特征结束日期 + 7天），例如 '2017-12-07'
    返回:
        pyspark.sql.DataFrame: 包含 user_id, item_id 和 label (0或1) 的标准样本标签表
    """
    print(f"【样本打标】开始切分时间窗。特征期截止: {feature_end_date}, 标签期截止: {label_end_date}")
    
    # 1. 提取特征期内活跃的所有“用户-商品对”作为我们的基础样本池（候选集）
    # 只有在特征期内与商品有过交互（浏览/加购/收藏/购买）的用户，才进入预测候选
    base_pairs = cleaned_df \
        .filter(col("date") <= feature_end_date) \
        .select("user_id", "item_id") \
        .distinct()
        
    # 2. 提取标签期（未来7天）内真正发生了购买行为（behavior_type == 'buy'）的记录
    true_buys = cleaned_df \
        .filter((col("date") > feature_end_date) & (col("date") <= label_end_date)) \
        .filter(col("behavior_type") == "buy") \
        .select("user_id", "item_id") \
        .distinct() \
        .withColumn("has_bought", lit(1))  # 标记为 1

    # 3. 将基础样本池与实际购买记录进行左外连接（Left Join）
    # 如果未来7天买了，has_bought 就是 1；如果没有买，has_bought 就是 Null
    joined_df = base_pairs.join(true_buys, on=["user_id", "item_id"], how="left")
    
    # 4. 将 Null 值填充为 0，生成最终的二分类标签（Label）
    # 1 代表未来7天购买了该商品，0 代表未购买
    final_labels_df = joined_df.withColumn(
        "label", 
        when(col("has_bought") == 1, 1).otherwise(0)
    ).drop("has_bought")
    
    print("【样本打标】打标完成！成功构建了包含正负样本的训练标签基础表。")
    return final_labels_df

if __name__ == "__main__":
    # 本地单独测试运行逻辑
    print("【本地测试】正在初始化测试用 SparkSession...")
    spark_test = SparkSession.builder \
        .master("local[*]") \
        .appName("LocalTestLabeling") \
        .getOrCreate()
        
    try:
        # 1. 模拟一个微型的清洗后数据集
        # 假设 2017-11-30 之前是特征期，2017-12-01 是未来的一天（标签期）
        from pyspark.sql.types import StructType, StructField, LongType, StringType
        
        schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("item_id", LongType(), True),
            StructField("behavior_type", StringType(), True),
            StructField("date_str", StringType(), True)
        ])
        
        mock_data = [
            (1001, 5001, "pv", "2017-11-29"),   # 用户1001在特征期看了商品5001
            (1001, 5001, "buy", "2017-12-01"),  # 用户1001在未来标签期买了商品5001 -> 应为正样本(1)
            (1002, 5002, "cart", "2017-11-30"), # 用户1002在特征期加购了商品5002
                                                # 用户1002在未来标签期没买商品5002 -> 应为负样本(0)
        ]
        
        mock_df = spark_test.createDataFrame(mock_data, schema=schema) \
            .withColumn("date", col("date_str").cast("date"))
            
        # 2. 运行打标函数
        labels_df = generate_user_item_labels(
            mock_df, 
            feature_end_date="2017-11-30", 
            label_end_date="2017-12-07"
        )
        
        # 3. 打印测试结果，验证是否符合预期
        labels_df.show()
        
    except Exception as e:
        print(f"【测试失败】错误信息: {e}")
    finally:
        spark_test.stop()