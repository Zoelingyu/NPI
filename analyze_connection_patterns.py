"""
analyze_connection_patterns_with_high_intensity.py
分析有效连接矩阵的连接模式：
1. 兴奋性连接和抑制性连接各自主要是网络内部还是网络之间
2. 兴奋性连接和抑制性连接各自主要是半球内部还是半球之间
3. 高强度连接（前1%）的网络/半球分布模式
"""

import numpy as np
import pandas as pd


def write_to_file(output_file, content):
    """将内容写入文件"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"写入文件失败: {e}")


def analyze_high_intensity_connections(matrix, region_info, top_percent=1, output_file=None):
    """
    分析高强度连接的网络/半球分布

    参数:
    ----------
    matrix : numpy array
        连接矩阵
    region_info : pandas DataFrame
        脑区信息
    top_percent : float
        高强度连接的百分比阈值
    output_file : str, optional
        输出文件路径
    """
    print(f"\n高强度连接分析（前{top_percent}%的连接）:")
    print("-" * 50)

    if output_file:
        write_to_file(output_file, f"\n\n高强度连接分析（前{top_percent}%的连接）:\n")
        write_to_file(output_file, "-" * 50 + "\n")

    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    row_indices, col_indices = np.where(mask)
    all_values = matrix[mask]

    # 分别处理兴奋性和抑制性连接
    for conn_type, type_name, sort_order in [
        ("excitatory", "兴奋性", -1),  # 降序，取最大正值
        ("inhibitory", "抑制性", 1)  # 升序，取最小负值（绝对值最大）
    ]:

        if conn_type == "excitatory":
            # 兴奋性连接：取正值
            conn_mask = all_values > 0
        else:
            # 抑制性连接：取负值
            conn_mask = all_values < 0

        if not np.any(conn_mask):
            print(f"\n   {type_name}连接: 无连接")
            continue

        # 提取连接
        conn_rows = row_indices[conn_mask]
        conn_cols = col_indices[conn_mask]
        conn_values = all_values[conn_mask]

        # 排序
        if conn_type == "excitatory":
            # 兴奋性：按值降序
            sorted_idx = np.argsort(conn_values)[::-1]
        else:
            # 抑制性：按绝对值降序（即按值升序，因为都是负数）
            sorted_idx = np.argsort(np.abs(conn_values))[::-1]

        # 取前top_percent%
        n_conn = len(conn_values)
        n_top = int(np.ceil(n_conn * top_percent / 100))

        if n_top == 0:
            print(f"\n   {type_name}连接: 高强度连接数为0")
            continue

        # 获取高强度连接的索引
        top_idx = sorted_idx[:n_top]
        top_rows = conn_rows[top_idx]
        top_cols = conn_cols[top_idx]
        top_values = conn_values[top_idx]

        # 高强度连接统计
        mean_strength = np.mean(np.abs(top_values))
        max_strength = np.max(np.abs(top_values)) if conn_type == "excitatory" else np.abs(np.min(top_values))
        min_strength = np.min(np.abs(top_values))

        print(f"\n   {type_name}连接高强度部分（前{top_percent}% = {n_top}/{n_conn}个连接）:")
        print(f"     强度范围: {min_strength:.4f} ~ {max_strength:.4f}")
        print(f"     平均强度: {mean_strength:.4f}")
        print(f"     总强度占比: {np.sum(np.abs(top_values)) / np.sum(np.abs(conn_values)) * 100:.1f}%")

        # 计算网络属性
        network_rows = region_info['network_id'].values[top_rows]
        network_cols = region_info['network_id'].values[top_cols]
        same_network = network_rows == network_cols

        # 计算半球属性
        hemi_rows = region_info['hemisphere'].values[top_rows]
        hemi_cols = region_info['hemisphere'].values[top_cols]
        same_hemisphere = hemi_rows == hemi_cols

        # 统计
        n_within_network = np.sum(same_network)
        n_between_network = n_top - n_within_network

        n_within_hemisphere = np.sum(same_hemisphere)
        n_between_hemisphere = n_top - n_within_hemisphere

        # 计算比例
        p_within_network = n_within_network / n_top * 100
        p_between_network = n_between_network / n_top * 100

        p_within_hemisphere = n_within_hemisphere / n_top * 100
        p_between_hemisphere = n_between_hemisphere / n_top * 100

        print(f"     网络内连接: {n_within_network:,} ({p_within_network:.1f}%)")
        print(f"     网络间连接: {n_between_network:,} ({p_between_network:.1f}%)")
        print(f"     半球内连接: {n_within_hemisphere:,} ({p_within_hemisphere:.1f}%)")
        print(f"     半球间连接: {n_between_hemisphere:,} ({p_between_hemisphere:.1f}%)")

        # 判断主要模式
        if p_within_network > 60:
            network_pattern = "主要在网络内部"
        elif p_between_network > 60:
            network_pattern = "主要在网络之间"
        else:
            network_pattern = "在网络内/间分布相对平衡"

        if p_within_hemisphere > 60:
            hemisphere_pattern = "主要在半球内部"
        elif p_between_hemisphere > 60:
            hemisphere_pattern = "主要在半球之间"
        else:
            hemisphere_pattern = "在半球内/间分布相对平衡"

        print(f"     → 高强度连接{network_pattern}")
        print(f"     → 高强度连接{hemisphere_pattern}")

        # 写入文件
        if output_file:
            file_content = f"""
{type_name}连接高强度部分（前{top_percent}% = {n_top}/{n_conn}个连接）:
  强度范围: {min_strength:.4f} ~ {max_strength:.4f}
  平均强度: {mean_strength:.4f}
  总强度占比: {np.sum(np.abs(top_values)) / np.sum(np.abs(conn_values)) * 100:.1f}%
  网络内连接: {n_within_network:,} ({p_within_network:.1f}%)
  网络间连接: {n_between_network:,} ({p_between_network:.1f}%)
  半球内连接: {n_within_hemisphere:,} ({p_within_hemisphere:.1f}%)
  半球间连接: {n_between_hemisphere:,} ({p_between_hemisphere:.1f}%)
  → 高强度连接{network_pattern}
  → 高强度连接{hemisphere_pattern}
"""
            write_to_file(output_file, file_content)


def analyze_connection_patterns(matrix_path, region_info_path, top_percent=1, output_file=None):
    """
    分析连接模式：网络内/网络间、半球内/半球间

    参数:
    ----------
    matrix_path : str
        连接矩阵文件路径
    region_info_path : str
        脑区信息文件路径
    top_percent : float
        高强度连接的百分比阈值
    output_file : str, optional
        输出文件路径
    """

    print("=" * 70)
    print("连接模式分析")
    print("=" * 70)

    if output_file:
        write_to_file(output_file, "\n" + "=" * 70 + "\n")
        write_to_file(output_file, "连接模式分析\n")
        write_to_file(output_file, "=" * 70 + "\n")

    # 1. 加载连接矩阵
    print("1. 加载数据...")
    try:
        matrix = np.load(matrix_path)
        print(f"   连接矩阵形状: {matrix.shape}")
    except Exception as e:
        print(f"   加载连接矩阵失败: {e}")
        return

    # 2. 加载脑区信息
    try:
        region_info = pd.read_csv(region_info_path)
        print(f"   脑区信息: {len(region_info)} 个脑区")
    except Exception as e:
        print(f"   加载脑区信息失败: {e}")
        return

    # 确保脑区数量匹配
    n_regions = matrix.shape[0]
    if len(region_info) != n_regions:
        print(f"   警告: 脑区数量不匹配 (矩阵: {n_regions}, 脑区信息: {len(region_info)})")
        return

    # 3. 准备脑区属性
    # 脑区编号（从1开始，转换为从0开始）
    region_info['EC_idx'] = region_info['EC_row'] - 1

    # 半球信息：奇数为左半球(1)，偶数为右半球(2)
    region_info['hemisphere'] = np.where(
        region_info['Label'] % 2 == 1, 1, 2
    )  # 1=左半球, 2=右半球

    # 网络信息
    network_names = region_info['network_name'].unique()
    print(f"   网络数量: {len(network_names)}")
    print(f"   网络名称: {', '.join(network_names)}")

    # 创建网络编号映射
    network_map = {name: idx + 1 for idx, name in enumerate(network_names)}
    region_info['network_id'] = region_info['network_name'].map(network_map)

    # 4. 分析连接模式
    print("\n2. 分析连接模式...")

    # 提取非对角线元素（排除自连接）
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    row_indices, col_indices = np.where(mask)

    # 兴奋性连接和抑制性连接
    excitatory_mask = matrix[mask] > 0
    inhibitory_mask = matrix[mask] < 0

    # 网络内/网络间分析
    print("\n3. 网络内/网络间连接分析:")

    for conn_type, type_name in [
        (excitatory_mask, "兴奋性连接"),
        (inhibitory_mask, "抑制性连接")
    ]:
        if not np.any(conn_type):
            print(f"\n   {type_name}: 无连接")
            continue

        # 获取连接对应的脑区对
        conn_rows = row_indices[conn_type]
        conn_cols = col_indices[conn_type]

        # 计算网络属性
        network_rows = region_info['network_id'].values[conn_rows]
        network_cols = region_info['network_id'].values[conn_cols]
        same_network = network_rows == network_cols

        # 计算半球属性
        hemi_rows = region_info['hemisphere'].values[conn_rows]
        hemi_cols = region_info['hemisphere'].values[conn_cols]
        same_hemisphere = hemi_rows == hemi_cols

        # 统计
        n_conn = len(conn_rows)
        n_within_network = np.sum(same_network)
        n_between_network = n_conn - n_within_network

        n_within_hemisphere = np.sum(same_hemisphere)
        n_between_hemisphere = n_conn - n_within_hemisphere

        # 计算比例
        p_within_network = n_within_network / n_conn * 100
        p_between_network = n_between_network / n_conn * 100

        p_within_hemisphere = n_within_hemisphere / n_conn * 100
        p_between_hemisphere = n_between_hemisphere / n_conn * 100

        print(f"\n   {type_name}:")
        print(f"     总数: {n_conn:,}")
        print(f"     网络内连接: {n_within_network:,} ({p_within_network:.1f}%)")
        print(f"     网络间连接: {n_between_network:,} ({p_between_network:.1f}%)")
        print(f"     半球内连接: {n_within_hemisphere:,} ({p_within_hemisphere:.1f}%)")
        print(f"     半球间连接: {n_between_hemisphere:,} ({p_between_hemisphere:.1f}%)")

        # 判断主要模式
        if p_within_network > 60:
            network_pattern = "主要是网络内部连接"
        elif p_between_network > 60:
            network_pattern = "主要是网络之间连接"
        else:
            network_pattern = "网络内/间连接相对平衡"

        if p_within_hemisphere > 60:
            hemisphere_pattern = "主要是半球内部连接"
        elif p_between_hemisphere > 60:
            hemisphere_pattern = "主要是半球之间连接"
        else:
            hemisphere_pattern = "半球内/间连接相对平衡"

        print(f"     → {network_pattern}")
        print(f"     → {hemisphere_pattern}")

        # 写入文件
        if output_file:
            file_content = f"""
{type_name}:
  总数: {n_conn:,}
  网络内连接: {n_within_network:,} ({p_within_network:.1f}%)
  网络间连接: {n_between_network:,} ({p_between_network:.1f}%)
  半球内连接: {n_within_hemisphere:,} ({p_within_hemisphere:.1f}%)
  半球间连接: {n_between_hemisphere:,} ({p_between_hemisphere:.1f}%)
  → {network_pattern}
  → {hemisphere_pattern}
"""
            write_to_file(output_file, file_content)

    # 5. 高强度连接分析
    analyze_high_intensity_connections(matrix, region_info, top_percent, output_file)

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


def main():
    """主函数"""
    # 文件路径
    matrix_path = r"...\ECmatrix_plcb_reordered_withSub.npy"
    region_info_path = r"...\sorted_brainnetome_yeo7network_withSubcortical.csv"

    # 输出文件路径
    output_file = r"...\ECdata_distribution_plcb.txt"

    # 设置高强度连接百分比阈值
    top_percent = 1  # 前1%的连接

    # 分析连接模式
    analyze_connection_patterns(matrix_path, region_info_path, top_percent, output_file)

    print(f"\n主要结果已追加到文件: {output_file}")


if __name__ == "__main__":
    main()