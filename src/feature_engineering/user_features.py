# src/feature_engineering/user_features.py
import pandas as pd
import numpy as np

def extract_user_base_features(behavior_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于用户行为数据集，提取用户维度(User)的核心特征。
    涵盖：活跃度、行为倾向性、购买转化率等。
    
    参数:
        behavior_df: 读入的淘宝用户行为 Pandas DataFrame
    返回:
        pd.DataFrame: 用户特征大表，主键为 user_id
    """
    print("【特征工程】开始提取用户维度(User)基础特征...")
    
    # 1. 提取每个用户在观测期内的总交互频次（活跃度）
    user_active_count = behavior_df.groupby("user_id").size().reset_index(name="u_total_actions")
    
    # 2. 统计用户浏览过的独立商品数与独立类目数（探索广度）
    user_unique_items = behavior_df.groupby("user_id").agg({
        "item_id": "nunique",
        "item_category": "nunique"
    }).reset_index().rename(columns={
        "item_id": "u_unique_items_count",
        "item_category": "u_unique_categories_count"
    })
    
    # 3. 细分行为统计：统计用户每种具体行为（点击、加购等）的次数
    # 假设：1=pv, 2=fav, 3=cart, 4=buy
    behavior_counts = behavior_df.groupby(["user_id", "behavior_type"]).size().unstack(fill_value=0)
    # 重命名列名，防止后续混淆
    behavior_counts = behavior_counts.rename(columns={
        1: "u_pv_count",
        2: "u_fav_count",
        3: "u_cart_count",
        4: "u_buy_count"
    }).reset_index()
    
    # 4. 融合各模块特征
    user_features = pd.merge(user_active_count, user_unique_items, on="user_id", how="left")
    user_features = pd.merge(user_features, behavior_counts, on="user_id", how="left")
    
    # 5. 计算衍生业务特征：购买转化率与加购傾向
    # 用户点击转化率 = 购买次数 / 总点击次数（加个小微值 0.001 防止分母为0）
    user_features["u_pv_to_buy_rate"] = user_features["u_buy_count"] / (user_features["u_pv_count"] + 0.001)
    
    # 用户深度交互倾向 = (加购+收藏) / 点击数
    user_features["u_deep_engagement_rate"] = (user_features["u_cart_count"] + user_features["u_fav_count"]) / (user_features["u_pv_count"] + 0.001)
    
    print(f"【特征工程】用户维度特征提取完成！共生成 {user_features.shape[1]-1} 个用户特征。")
    return user_features

if __name__ == "__main__":
    # 本地小样本运行与自测
    print("【本地测试】正在加载部分用户数据...")
    try:
        # 读取前10000条数据进行快速特征抽取验证
        df_sample = pd.read_csv("taobao_user_behavior_processed.xlsx - taobao_user_behavior_processed.csv", nrows=10000)
        
        # 运行特征提取
        u_feats = extract_user_base_features(df_sample)
        
        # 打印特征矩阵的部分结果和空值率验证
        print("\n--- 提取出的用户特征预览 (前5行) ---")
        print(u_feats.head())
        
        print("\n--- 特征质量检查（缺失值统计） ---")
        print(u_feats.isnull().sum())
        
    except Exception as e:
        print(f"【测试失败】未找到数据集或代码存在逻辑错误，报错信息: {e}")