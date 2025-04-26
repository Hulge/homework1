import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import io
from collections import Counter


# ======== 中文输出环境配置 ========
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======== 累积容器 ========
category_list = []
country_list = []
price_by_category = {}
income_list = []
login_count_list = []
avg_session_duration_list = []


# ======== 文件目录与参数 ========
data_dir = "data/raw data/10G_data_new"
BATCH_SIZE = 3

parquet_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".parquet")
])


# ======== 遍历 Parquet 文件 ========
for file_path in parquet_files:
    pf = pq.ParquetFile(file_path)
    print(f"正在处理文件: {file_path}，共 {pf.num_row_groups} 个 row groups")

    for start in range(0, pf.num_row_groups, BATCH_SIZE):
        end = min(start + BATCH_SIZE, pf.num_row_groups)
        print(f"批处理 row groups: {start} - {end - 1}")
        tables = [pf.read_row_group(i) for i in range(start, end)]
        combined_table = pa.concat_tables(tables)
        df = combined_table.to_pandas()

        # JSON 字段解析
        def safe_json(text):
            try:
                return json.loads(text)
            except:
                return {}

        df['purchase_history'] = df['purchase_history'].apply(safe_json)
        df['login_history'] = df['login_history'].apply(safe_json)

        df['category'] = df['purchase_history'].apply(lambda x: x.get('categories'))
        df['average_price'] = df['purchase_history'].apply(lambda x: x.get('avg_price'))
        df['login_count'] = df['login_history'].apply(lambda x: x.get('login_count'))
        df['avg_session_duration'] = df['login_history'].apply(lambda x: x.get('avg_session_duration'))

        category_list.extend(df['category'])
        country_list.extend(df['country'])
        income_list.extend(df['income'].dropna())
        login_count_list.extend(df['login_count'].dropna())
        avg_session_duration_list.extend(df['avg_session_duration'].dropna())

        for cat, price in zip(df['category'], df['average_price']):
            if cat not in price_by_category:
                price_by_category[cat] = []
            price_by_category[cat].append(price)


# ===================== 可视化分析 =====================

# ======== 用户收入分布直方图 ========
plt.figure(figsize=(8, 6))
sns.histplot(income_list, bins=30, kde=True)
plt.title("用户收入分布")
plt.xlabel("收入")
plt.ylabel("人数")
plt.grid(True)
plt.tight_layout()
plt.show()

# ======== 商品类别消费比例饼图 ========
plt.figure(figsize=(6, 6))
cat_counts = Counter(category_list)
plt.pie(cat_counts.values(), labels=cat_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title("商品类别消费比例")
plt.axis('equal')
plt.show()

# ======== 国家用户数量分布柱状图 ========
plt.figure(figsize=(10, 5))
country_counts = Counter(country_list)
sns.barplot(x=list(country_counts.keys()), y=list(country_counts.values()))
plt.title("各国用户数量分布")
plt.xlabel("国家")
plt.ylabel("用户数量")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======== 商品类别的平均消费金额箱线图 ========
boxplot_df = pd.DataFrame([
    {"category": cat, "average_price": price}
    for cat, prices in price_by_category.items()
    for price in prices
])
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='average_price', data=boxplot_df)
plt.title("不同商品类别的平均消费金额分布")
plt.xlabel("商品类别")
plt.ylabel("平均消费金额")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======== 登录次数 vs 平均单次时长散点图 ========
plt.figure(figsize=(8, 6))
sns.scatterplot(x=login_count_list, y=avg_session_duration_list)
plt.title("登录次数与平均单次登录时长关系")
plt.xlabel("登录次数")
plt.ylabel("平均单次登录时长（分钟）")
plt.grid(True)
plt.tight_layout()
plt.show()