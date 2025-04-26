import os
import json
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import io
import gc


# ======== 中文输出环境配置 ========
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======== 工具函数 ========
def safe_json(text):
    try:
        return json.loads(text) if isinstance(text, str) else text
    except Exception:
        return {}

def validate_logical_consistency(row):
    reg = row['registration_date']
    last_login = row['last_login']
    login_first = row.get('login_first')

    errors = []
    if pd.notna(reg) and pd.notna(login_first) and login_first < reg:
        errors.append("首登时间早于注册")
    if pd.notna(reg) and pd.notna(last_login) and last_login < reg:
        errors.append("最后登入早于注册")
    return "; ".join(errors) if errors else None


# ======== 主流程开始 ========
print("启动数据预处理任务...")
data_dir = "data/raw data/30G_data_new"
output_dir = "data/preprocessed data/30G_data_preprocessed"
BATCH_SIZE = 3

os.makedirs(output_dir, exist_ok=True)

parquet_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".parquet")
])
print(f"找到 {len(parquet_files)} 个 Parquet 文件")


# 用于全局统计
global_missing = {}
global_logic_errors = 0
global_logic_error_first_login = 0
global_logic_error_last_login = 0
global_total_records = 0


for file_index, file_path in enumerate(parquet_files):
    print(f"\n读取文件 {file_index + 1}/{len(parquet_files)}: {file_path}")
    pf = pq.ParquetFile(file_path)
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    writer = None

    for start in range(0, pf.num_row_groups, BATCH_SIZE):
        print(f"------------处理 row group {start} 到 {min(start + BATCH_SIZE, pf.num_row_groups) - 1}------------")
        tables = [pf.read_row_group(i) for i in range(start, min(start + BATCH_SIZE, pf.num_row_groups))]
        df = pa.concat_tables(tables).to_pandas()
        print(f"载入 DataFrame，记录数: {len(df)}")

        print("替换缺失地址字段...")
        df['address'] = df['address'].replace({'': None, 'Non-Chinese Address Placeholder': None})

        print("转换时间字段格式...")
        df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce').dt.tz_localize(None)
        df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce').dt.tz_localize(None)

        print("解析 login_history 和 purchase_history 字段...")
        df['login_history'] = df['login_history'].apply(safe_json)
        df['purchase_history'] = df['purchase_history'].apply(safe_json)

        df['purchase_avg_price'] = df['purchase_history'].apply(lambda x: x.get('avg_price'))
        df['purchase_categories'] = df['purchase_history'].apply(lambda x: x.get('categories'))
        df['purchase_items'] = df['purchase_history'].apply(lambda x: x.get('items'))
        df['purchase_payment_method'] = df['purchase_history'].apply(lambda x: x.get('payment_method'))
        df['purchase_payment_status'] = df['purchase_history'].apply(lambda x: x.get('payment_status'))
        df['purchase_date'] = pd.to_datetime(df['purchase_history'].apply(lambda x: x.get('purchase_date')), errors='coerce').dt.tz_localize(None)

        df['login_avg_session_duration'] = df['login_history'].apply(lambda x: x.get('avg_session_duration'))
        df['login_devices'] = df['login_history'].apply(lambda x: x.get('devices'))
        df['login_locations'] = df['login_history'].apply(lambda x: x.get('locations'))
        df['login_count'] = df['login_history'].apply(lambda x: x.get('login_count'))
        df['login_first'] = pd.to_datetime(df['login_history'].apply(lambda x: x.get('first_login')), errors='coerce').dt.tz_localize(None)
        df['login_timestamps'] = df['login_history'].apply(lambda x: x.get('timestamps'))

        df.drop(columns=['login_history', 'purchase_history'], inplace=True)

        print("校验字段逻辑一致性...")
        df['consistency_errors'] = df.apply(validate_logical_consistency, axis=1)


        # 缺失数据统计
        missing_summary = df.drop(columns=['consistency_errors'], errors='ignore').isnull().sum()
        total_records = len(df)
        global_total_records += total_records
        for col, count in missing_summary.items():
            global_missing[col] = global_missing.get(col, 0) + count
        print("缺失数据统计:", end=' ')
        summary_line = []
        for col, missing_count in missing_summary[missing_summary > 0].items():
            ratio = missing_count / total_records
            summary_line.append(f"{col}: {missing_count} ({ratio:.2%})")
        print(" | ".join(summary_line))


        # 逻辑错误记录统计
        logic_errors = df['consistency_errors'].notnull().sum()
        logic_error_first_login = df['consistency_errors'].str.contains("首登时间早于注册", na=False).sum()
        logic_error_last_login = df['consistency_errors'].str.contains("最后登入早于注册", na=False).sum()

        global_logic_errors += logic_errors
        global_logic_error_first_login += logic_error_first_login
        global_logic_error_last_login += logic_error_last_login

        print(f"存在逻辑错误的记录数：{logic_errors} ({logic_errors / total_records:.2%})")
        print(f"└─ 首登时间早于注册：{logic_error_first_login} ({logic_error_first_login / total_records:.2%})")
        print(f"└─ 最后登入早于注册：{logic_error_last_login} ({logic_error_last_login / total_records:.2%})")


        # 写入Parquet文件
        print("写入当前批次结果到 Parquet 文件...")
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

        del df, table, tables
        gc.collect()

    if writer:
        writer.close()
        print(f"已完成写入文件: {output_path}")


# ======== 最终统计输出 ========
print(f"\n全部数据记录总数：{global_total_records}")

print("全部数据缺失值统计:", end=' ')
summary_line = []
for col, count in global_missing.items():
    if count > 0:
        ratio = count / global_total_records
        summary_line.append(f"{col}: {count} ({ratio:.2%})")
print(" | ".join(summary_line))


print(f"全部数据逻辑错误记录数：{global_logic_errors} ({global_logic_errors / global_total_records:.2%})")
print(f"└─ 首登时间早于注册：{global_logic_error_first_login} ({global_logic_error_first_login / global_total_records:.2%})")
print(f"└─ 最后登入早于注册：{global_logic_error_last_login} ({global_logic_error_last_login / global_total_records:.2%})")

print("\n所有数据处理完毕，已写入目录：30G_data_preprocessed")