import numpy as np
import os
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm  # 导入进度条库

# 设置输出目录
output_dir = r"...\ttest_withSubcortical"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在


# 1. 数据加载与配对
def load_and_pair_data(lsd_dir, plcb_dir):
    """加载并配对LSD与安慰剂矩阵"""
    print("开始加载数据...")
    # 获取文件列表并提取被试ID
    lsd_files = [f for f in os.listdir(lsd_dir) if f.endswith('.npy')]
    plcb_files = [f for f in os.listdir(plcb_dir) if f.endswith('.npy')]

    # 创建被试ID映射
    subj_ids = [f.split('_')[1] for f in lsd_files]  # 文件名格式: lsd01_001_EC_reordered.npy

    # 初始化存储
    all_lsd = []
    all_plcb = []

    # 加载并配对数据
    print(f"配对 {len(subj_ids)} 名被试的数据...")
    for subj_id in tqdm(subj_ids, desc="加载被试数据"):
        lsd_path = os.path.join(lsd_dir, f"lsd01_{subj_id}_EC_reordered.npy")
        plcb_path = os.path.join(plcb_dir, f"plcb01_{subj_id}_EC_reordered.npy")

        if os.path.exists(lsd_path) and os.path.exists(plcb_path):
            lsd_matrix = np.load(lsd_path)
            plcb_matrix = np.load(plcb_path)
            all_lsd.append(lsd_matrix)
            all_plcb.append(plcb_matrix)

    return np.array(all_lsd), np.array(all_plcb), subj_ids


# 2. 配对t检验分析（添加进度条）
def paired_t_test_analysis(lsd_data, plcb_data):
    """执行体素级配对t检验"""
    print("执行配对t检验...")
    # 计算差异矩阵
    diff_mats = lsd_data - plcb_data

    # 初始化结果存储
    t_map = np.zeros_like(lsd_data[0])
    p_map = np.zeros_like(lsd_data[0])

    # 获取矩阵维度
    n_rois = lsd_data.shape[1]
    total_pairs = n_rois * n_rois

    # 逐体素进行配对t检验（添加进度条）
    with tqdm(total=total_pairs, desc="计算连接差异") as pbar:
        for i in range(n_rois):
            for j in range(n_rois):
                t, p = stats.ttest_rel(lsd_data[:, i, j], plcb_data[:, i, j])
                t_map[i, j] = t
                p_map[i, j] = p
                pbar.update(1)  # 更新进度条

    return t_map, p_map


# 3. 多重比较校正
def correct_multiple_comparisons(p_map, method='fdr'):
    """应用多重比较校正"""
    print("执行多重比较校正...")
    # 将p值展平为一维
    p_flat = p_map.flatten()

    # FDR校正
    if method == 'fdr':
        reject, p_corrected = fdrcorrection(p_flat, alpha=0.05)

    # 还原为矩阵
    p_corrected_map = p_corrected.reshape(p_map.shape)
    sig_map = reject.reshape(p_map.shape)

    return p_corrected_map, sig_map


# 4. 置换检验验证（添加进度条）
def permutation_test(lsd_data, plcb_data, n_perm=5000):
    """置换检验验证显著性"""
    print(f"执行置换检验 ({n_perm}次置换)...")
    n_subj = lsd_data.shape[0]
    n_rois = lsd_data.shape[1]

    # 初始化置换分布
    perm_t_max = np.zeros(n_perm)

    # 原始统计量
    orig_t_map, _ = paired_t_test_analysis(lsd_data, plcb_data)
    orig_t_max = np.max(np.abs(orig_t_map))

    # 执行置换（添加进度条）
    for p in tqdm(range(n_perm), desc="置换检验"):
        # 随机翻转条件标签
        perm_data = np.copy(lsd_data)
        for s in range(n_subj):
            if np.random.rand() > 0.5:
                perm_data[s] = plcb_data[s]  # 交换条件

        # 计算置换后统计量
        perm_t_map, _ = paired_t_test_analysis(perm_data, plcb_data)
        perm_t_max[p] = np.max(np.abs(perm_t_map))

    # 计算p值
    p_value = np.sum(perm_t_max >= orig_t_max) / n_perm

    return p_value, perm_t_max


# 主流程
if __name__ == "__main__":
    # 设置路径
    lsd_dir = r'...\EC_reordered_withSubcortical\lsd'
    plcb_dir = r'...\EC_reordered_withSubcortical\plcb'

    # 1. 加载并配对数据
    lsd_data, plcb_data, subj_ids = load_and_pair_data(lsd_dir, plcb_dir)
    print(f"成功加载 {len(subj_ids)} 名被试的数据")

    # 2. 执行配对t检验
    t_map, p_map = paired_t_test_analysis(lsd_data, plcb_data)

    # 3. FDR校正
    p_corrected_map, sig_map = correct_multiple_comparisons(p_map, method='fdr')

    # 4. 置换检验验证
    perm_p, perm_dist = permutation_test(lsd_data, plcb_data, n_perm=5000)
    print(f"置换检验p值: {perm_p:.4f}")

    # 5. 保存统计结果
    print(f"保存结果到 {output_dir}")
    np.save(os.path.join(output_dir, 't_map.npy'), t_map)
    np.save(os.path.join(output_dir, 'p_map.npy'), p_map)
    np.save(os.path.join(output_dir, 'p_corrected_map.npy'), p_corrected_map)
    np.save(os.path.join(output_dir, 'sig_map.npy'), sig_map)

    # 6. 生成结果报告
    n_significant = np.sum(sig_map)
    print(f"发现 {n_significant} 个显著差异连接 (FDR<0.05)")

    # 7. 保存置换检验结果
    np.save(os.path.join(output_dir, 'perm_dist.npy'), perm_dist)
    with open(os.path.join(output_dir, 'perm_test_results.txt'), 'w') as f:
        f.write(f"置换检验p值: {perm_p:.4f}\n")
        f.write(f"显著差异连接数: {n_significant}\n")

    print("分析完成！")
