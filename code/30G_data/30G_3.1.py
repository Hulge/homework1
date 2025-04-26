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
income_list = []
age_list = []
is_active_list = []
category_list = []
country_list = []
price_by_category = {}


# ======== 文件目录与参数 ========
data_dir = "data/raw data/30G_data_new"
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

        df['category'] = df['purchase_history'].apply(lambda x: x.get('categories'))
        df['average_price'] = df['purchase_history'].apply(lambda x: x.get('avg_price'))

        income_list.extend(df['income'].dropna())
        age_list.extend(df['age'].dropna())
        is_active_list.extend(df['is_active'].dropna())
        category_list.extend(df['category'])
        country_list.extend(df['country'])

        for cat, price in zip(df['category'], df['average_price']):
            if cat not in price_by_category:
                price_by_category[cat] = []
            price_by_category[cat].append(price)


# ===================== 可视化分析 =====================

# ======== 用户年龄分布直方图 ========
plt.figure(figsize=(8, 6))
sns.histplot(age_list, bins=30, kde=True)
plt.title("用户年龄分布")
plt.xlabel("年龄")
plt.ylabel("人数")
plt.grid(True)
plt.tight_layout()
plt.show()

# ======== 不同商品类别消费金额小提琴图 ========
top5_categories = [cat for cat, _ in Counter(category_list).most_common(5)]
filtered_prices = [
    {"category": cat, "average_price": price}
    for cat, prices in price_by_category.items() if cat in top5_categories
    for price in prices
]
top5_df = pd.DataFrame(filtered_prices)

plt.figure(figsize=(12, 6))
sns.violinplot(x='category', y='average_price', data=top5_df)
plt.title("前五大商品类别消费金额分布")
plt.xlabel("商品类别")
plt.ylabel("消费金额")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======== 活跃与非活跃用户比例饼图 ========
plt.figure(figsize=(6, 6))
active_counts = Counter(is_active_list)
labels = ["活跃", "非活跃"]
plt.pie(active_counts.values(), labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("活跃与非活跃用户比例")
plt.axis('equal')
plt.show()

# ======== 各国用户平均收入柱状图 ========
plt.figure(figsize=(10, 6))
country_income_df = pd.DataFrame({'country': country_list, 'income': income_list})
mean_income_by_country = country_income_df.groupby('country')['income'].mean().sort_values(ascending=False)
sns.barplot(x=mean_income_by_country.index, y=mean_income_by_country.values)
plt.title("各国用户平均收入")
plt.xlabel("国家")
plt.ylabel("平均收入")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()