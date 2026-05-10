import numpy as np
import pandas as pd
import os
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# ============================
# 1. 数据准备
# ============================
# 设置路径
base_dir = Path(r"...\EC_reordered_withSubcortical")
lsd_dir = base_dir / "lsd"
plcb_dir = base_dir / "plcb"
output_dir = r'...\top_features\region'
os.makedirs(output_dir, exist_ok=True)
meta_path = Path(
    r"...\brainnetome\sorted_brainnetome_yeo7network_withSubcortical.csv")

# 加载脑区元数据
meta_df = pd.read_csv(meta_path)
n_regions = 246

# 创建从索引到脑区信息的映射
region_info = {}
for idx, row in meta_df.iterrows():
    ec_index = int(row['EC_row'])  # 脑区在EC矩阵中的索引
    region_info[ec_index] = {
        'name': row['subregion_name'],
        'network_id': int(row['Yeo_7network_id']),
        'network_name': row['network_name']
    }

# 创建EC_row到Label的映射
ec_row_to_label = {}
for idx, row in meta_df.iterrows():
    ec_row = int(row['EC_row'])
    label = row['Label']
    ec_row_to_label[ec_row] = label

# 加载被试数据
subject_ids = sorted([f.name.split('_')[1] for f in lsd_dir.glob("*.npy")])
print(f"找到 {len(subject_ids)} 名被试")

# 初始化数据矩阵
n_subjects = len(subject_ids)
X = np.zeros((n_subjects * 2, n_regions, n_regions))  # 30个样本×246×246
y = np.zeros(n_subjects * 2)
groups = np.zeros(n_subjects * 2)  # 用于分组交叉验证

# 加载EC矩阵
for i, sub_id in enumerate(tqdm(subject_ids, desc="加载数据")):
    # LSD条件
    lsd_path = lsd_dir / f"lsd01_{sub_id}_EC_reordered.npy"
    lsd_ec = np.load(lsd_path)
    X[i * 2] = lsd_ec
    y[i * 2] = 1  # LSD标签

    # 安慰剂条件
    plcb_path = plcb_dir / f"plcb01_{sub_id}_EC_reordered.npy"
    plcb_ec = np.load(plcb_path)
    X[i * 2 + 1] = plcb_ec
    y[i * 2 + 1] = 0  # 安慰剂标签

    groups[i * 2] = i
    groups[i * 2 + 1] = i

print(f"数据矩阵形状: {X.shape}, 标签形状: {y.shape}")


# ============================
# 2. 脑区特征评估函数（修改版）
# ============================
def evaluate_region_features(X, y, groups, region_idx, feature_type):
    """
    评估单个脑区行或列作为特征时的分类性能
    :param X: 完整的EC矩阵 (样本数×脑区数×脑区数)
    :param y: 标签向量
    :param groups: 分组向量
    :param region_idx: 脑区索引 (0-based)
    :param feature_type: 'row' 或 'col'
    :return: 性能指标字典和平均系数向量
    """
    # 提取特征
    if feature_type == 'row':
        X_feat = X[:, region_idx, :]  # 提取该脑区的行特征
    else:  # 'col'
        X_feat = X[:, :, region_idx]  # 提取该脑区的列特征

    # 初始化模型和标准化器
    model = LogisticRegression(penalty='l2', max_iter=2000, solver='lbfgs')
    scaler = StandardScaler()
    group_kfold = GroupKFold(n_splits=5)

    # 初始化指标存储
    accuracies = []
    sensitivities = []
    specificities = []
    precisions = []
    aurocs = []

    # 初始化系数存储
    coefs_list = []  # 存储每个fold的系数

    # 分组交叉验证
    for train_idx, test_idx in group_kfold.split(X_feat, y, groups):
        # 训练集和测试集
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 保存系数
        coefs_list.append(model.coef_[0].copy())

        # 预测
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # 计算特异度
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 计算AUROC
        auroc = roc_auc_score(y_test, y_prob)

        # 存储指标
        accuracies.append(acc)
        sensitivities.append(recall)
        specificities.append(specificity)
        precisions.append(precision)
        aurocs.append(auroc)

    # 计算平均指标
    metrics = {
        'accuracy': np.mean(accuracies),
        'sensitivity': np.mean(sensitivities),
        'specificity': np.mean(specificities),
        'precision': np.mean(precisions),
        'auroc': np.mean(aurocs),
        'accuracy_std': np.std(accuracies),
        'sensitivity_std': np.std(sensitivities),
        'specificity_std': np.std(specificities),
        'precision_std': np.std(precisions),
        'auroc_std': np.std(aurocs)
    }

    # 计算平均系数
    avg_coefs = np.mean(coefs_list, axis=0)

    return metrics, avg_coefs


# ============================
# 3. 评估所有脑区的行和列特征（修改版）
# ============================
results = []
all_coefs = []  # 存储所有脑区特征的系数
all_coefs_info = []  # 存储系数对应的信息

# 进度条
pbar = tqdm(total=n_regions * 2, desc="评估脑区特征")

for region_idx in range(n_regions):
    # 获取脑区信息
    ec_row = region_idx + 1  # EC_row从1开始
    region_data = region_info.get(ec_row, {'name': f'Unknown_{ec_row}', 'network_name': 'Unknown'})
    region_name = region_data['name']
    network_name = region_data['network_name']

    # 获取Label_inBNA
    label_inBNA = ec_row_to_label.get(ec_row, 'Unknown')

    # 评估行特征
    row_metrics, row_coefs = evaluate_region_features(X, y, groups, region_idx, 'row')
    row_metrics.update({
        'region_index_inEC': ec_row,
        'region_name': region_name,
        'network_name': network_name,
        'feature_type': 'row',
        'Label_inBNA': label_inBNA  # 添加Label_inBNA
    })
    results.append(row_metrics)

    # 保存行特征系数
    all_coefs.append(row_coefs)
    all_coefs_info.append({
        'region_index_inEC': ec_row,
        'region_name': region_name,
        'feature_type': 'row',
        'Label_inBNA': label_inBNA
    })

    pbar.update(1)

    # 评估列特征
    col_metrics, col_coefs = evaluate_region_features(X, y, groups, region_idx, 'col')
    col_metrics.update({
        'region_index_inEC': ec_row,
        'region_name': region_name,
        'network_name': network_name,
        'feature_type': 'col',
        'Label_inBNA': label_inBNA  # 添加Label_inBNA
    })
    results.append(col_metrics)

    # 保存列特征系数
    all_coefs.append(col_coefs)
    all_coefs_info.append({
        'region_index_inEC': ec_row,
        'region_name': region_name,
        'feature_type': 'col',
        'Label_inBNA': label_inBNA
    })

    pbar.update(1)

pbar.close()

# ============================
# 4. 保存结果并打印前10脑区
# ============================
# 转换为DataFrame
results_df = pd.DataFrame(results)

# 保存结果
results_df.to_csv(os.path.join(output_dir, 'region_performance.csv'), index=False)
print(f"结果已保存至: {os.path.join(output_dir, 'region_performance.csv')}")

# 按AUROC降序排列后保存
sorted_results_df = results_df.sort_values('auroc', ascending=False)
sorted_results_df.to_csv(os.path.join(output_dir, 'sorted_region_performance.csv'), index=False)
print(f"排序结果已保存至: {os.path.join(output_dir, 'sorted_region_performance.csv')}")

# 打印前10个贡献最大的脑区（按AUROC降序）
top_regions = results_df.sort_values('auroc', ascending=False).head(20)

print("\n===== 前10个贡献最大的脑区 =====")
print("按AUROC降序排列:")
print(top_regions[
          ['region_index_inEC', 'region_name', 'network_name', 'feature_type', 'Label_inBNA', 'auroc', 'accuracy',
           'sensitivity', 'specificity',
           'precision']].to_string(index=False))

# 打印脑区名称
print("\n前10个脑区名称:")
for _, row in top_regions.iterrows():
    print(
        f"{row['region_name']} ({'行' if row['feature_type'] == 'row' else '列'}) - {row['network_name']} - Label: {row['Label_inBNA']} - AUROC: {row['auroc']:.4f}")

# ============================
# 5. 保存模型系数
# ============================
# 创建系数矩阵（每个脑区特征一行，246列）
coefs_matrix = np.array(all_coefs)
print(f"系数矩阵形状: {coefs_matrix.shape}")

# 创建系数信息DataFrame
coefs_info_df = pd.DataFrame(all_coefs_info)

# 保存系数矩阵和系数信息
coefs_output_path = os.path.join(output_dir, 'region_coefficients.csv')
coefs_matrix_df = pd.DataFrame(coefs_matrix)
coefs_matrix_df = pd.concat([coefs_info_df, coefs_matrix_df], axis=1)
coefs_matrix_df.to_csv(coefs_output_path, index=False)
print(f"所有脑区特征系数已保存至: {coefs_output_path}")

# 保存AUROC最高的脑区特征的系数
best_region = sorted_results_df.iloc[0]
best_region_idx = best_region['region_index_inEC']
best_feature_type = best_region['feature_type']

# 找到对应的系数
best_coefs = None
for i, info in enumerate(all_coefs_info):
    if info['region_index_inEC'] == best_region_idx and info['feature_type'] == best_feature_type:
        best_coefs = all_coefs[i]
        break

if best_coefs is not None:
    best_coefs_path = os.path.join(output_dir, f'best_region_coefs_{best_region_idx}_{best_feature_type}.npy')
    np.save(best_coefs_path, best_coefs)
    print(f"AUROC最高的脑区特征系数已保存至: {best_coefs_path}")

    # 同时保存为CSV
    best_coefs_df = pd.DataFrame({
        'brain_region': [f"{best_region['region_name']} ({best_feature_type})"],
        'region_index': [best_region_idx],
        'feature_type': [best_feature_type],
        'auroc': [best_region['auroc']]
    })
    for j in range(len(best_coefs)):
        best_coefs_df[f'coef_{j}'] = best_coefs[j]

    best_coefs_csv_path = os.path.join(output_dir, f'best_region_coefs_{best_region_idx}_{best_feature_type}.csv')
    best_coefs_df.to_csv(best_coefs_csv_path, index=False)
    print(f"AUROC最高的脑区特征系数(CSV)已保存至: {best_coefs_csv_path}")
else:
    print("警告：未找到AUROC最高的脑区特征系数")

print("分析完成！")