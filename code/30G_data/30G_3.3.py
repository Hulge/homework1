import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import sys
import io


# ======== 中文输出环境配置 ========
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======== 配置路径和参数 ========
preprocessed_dir = "data/preprocessed data/30G_data_preprocessed"
n_clusters = 5
batch_size = 50000
feature_columns = ['income', 'purchase_avg_price', 'login_count', 'login_avg_session_duration']


# ======== 初始化聚类器和标准化器 ========
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
scaler = StandardScaler()


# ======== 第一步：拟合标准化器 ========
print("正在收集数据以拟合标准化器...")
all_batches = []
for file in os.listdir(preprocessed_dir):
    if file.endswith(".parquet"):
        path = os.path.join(preprocessed_dir, file)
        df = pq.read_table(path).to_pandas()
        batch = df[feature_columns].dropna()
        if not batch.empty:
            all_batches.append(batch)

if not all_batches:
    raise ValueError("没有找到任何可用于拟合的数据")

all_data = pd.concat(all_batches, ignore_index=True)
scaler.fit(all_data)
print("标准化器拟合完成")


# ======== 第二步：逐文件增量聚类 ========
print("开始逐文件进行 MiniBatchKMeans 聚类...")
for file in os.listdir(preprocessed_dir):
    if file.endswith(".parquet"):
        path = os.path.join(preprocessed_dir, file)
        df = pq.read_table(path).to_pandas()
        batch = df[feature_columns].dropna()
        if batch.empty:
            continue
        X_scaled = scaler.transform(batch)
        kmeans.partial_fit(X_scaled)

print("聚类模型训练完成")


# ======== 输出聚类中心（反标准化） ========
print("\n聚类中心（反标准化后）：")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers):
    print(f"Cluster {i+1}: {dict(zip(feature_columns, center.round(2)))}")


# ======== 打标签并写入原始文件，同时记录每个 cluster 的成员 id 和 is_active ========
print("\n开始为每位用户分配 cluster_id 标签并覆盖原始文件...")

# 创建字典存储每个 cluster 对应的 (id, is_active) 列表
cluster_id_to_members = {}

for file in os.listdir(preprocessed_dir):
    if file.endswith(".parquet"):
        path = os.path.join(preprocessed_dir, file)
        df = pq.read_table(path).to_pandas()

        # 提取用于聚类的特征，并筛除缺失值
        feature_df = df[feature_columns]
        mask = feature_df.notnull().all(axis=1)
        X_scaled = scaler.transform(feature_df[mask])

        # 获取聚类标签
        cluster_labels = kmeans.predict(X_scaled) + 1

        # 初始化 cluster_id 列为 0（代表未聚类）
        df['cluster_id'] = 0
        df.loc[mask, 'cluster_id'] = cluster_labels

        # 收集每个簇下的 id 和 is_active
        valid_ids = df.loc[mask, 'id'].values
        valid_status = df.loc[mask, 'is_active'].values
        for cid, uid, active in zip(cluster_labels, valid_ids, valid_status):
            if cid not in cluster_id_to_members:
                cluster_id_to_members[cid] = []
            cluster_id_to_members[cid].append({'id': uid, 'is_active': active})

        # 覆盖写入原 parquet 文件
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        print(f"已更新文件并添加 cluster_id: {path}")


# ======== 写入每个聚类簇的 id + is_active 到独立 parquet 文件 ========
print("\n开始写入每个聚类簇对应的用户 id + is_active 到 parquet 文件...")
output_dir = "data/preprocessed data/cluster_data/30G_cluster_members"
os.makedirs(output_dir, exist_ok=True)

for cluster_id, member_list in cluster_id_to_members.items():
    cluster_df = pd.DataFrame(member_list)  # 每一项是 {'id': ..., 'is_active': ...}
    output_path = os.path.join(output_dir, f"cluster_{cluster_id}.parquet")
    table = pa.Table.from_pandas(cluster_df)
    pq.write_table(table, output_path)
    print(f"Cluster {cluster_id} 的用户 ID 和 is_active 状态已保存到: {output_path}")

print("\n所有用户均已添加 cluster_id，聚类簇成员已保存")