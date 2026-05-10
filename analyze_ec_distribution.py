"""
analyze_ec_all_distributions.py
有效连接矩阵分布分析 - 完整版
功能：
1. 判断绝对值连接是否呈长尾分布
2. 分别拟合兴奋性和抑制性连接的四种分布
3. 拟合所有连接（绝对值）的四种分布
"""

import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew, lognorm, norm, expon, invgauss
import warnings
import sys
import io

warnings.filterwarnings('ignore')


class TeeOutput:
    """将输出同时发送到控制台和文件的类"""

    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


def setup_output_redirection(log_file_path):
    """设置输出重定向，将print输出同时发送到控制台和文件"""
    # 保存原始的stdout
    original_stdout = sys.stdout

    # 打开日志文件
    log_file = open(log_file_path, 'w', encoding='utf-8')

    # 创建Tee输出对象
    tee = TeeOutput(original_stdout, log_file)

    # 重定向stdout
    sys.stdout = tee

    return original_stdout, log_file


def restore_output_redirection(original_stdout, log_file):
    """恢复原始的stdout并关闭日志文件"""
    sys.stdout = original_stdout
    if log_file:
        log_file.close()


def fit_distributions(data, data_type="数据"):
    """
    拟合四种分布并比较

    参数:
    ----------
    data : numpy array
        要拟合的数据
    data_type : str
        数据类型描述

    返回:
    ----------
    best_fit : str
        最佳拟合的分布名称
    results : dict
        所有分布的拟合结果
    """
    results = {}

    print(f"\n{data_type}分布拟合结果:")
    print("-" * 60)

    for dist_name in ["lognormal", "normal", "exponential", "inverse_gaussian"]:
        try:
            if dist_name == "lognormal":
                # 对数正态分布拟合
                shape, loc, scale = lognorm.fit(data, floc=0)
                params = (shape, loc, scale)
                dist = lognorm
            elif dist_name == "normal":
                # 正态分布拟合
                loc, scale = norm.fit(data)
                params = (loc, scale)
                dist = norm
            elif dist_name == "exponential":
                # 指数分布拟合
                loc, scale = expon.fit(data, floc=0)
                params = (loc, scale)
                dist = expon
            elif dist_name == "inverse_gaussian":
                # 逆高斯分布拟合
                mu, loc, scale = invgauss.fit(data, floc=0)
                params = (mu, loc, scale)
                dist = invgauss

            # 计算AIC
            log_likelihood = np.sum(dist.logpdf(data, *params))
            aic = 2 * len(params) - 2 * log_likelihood

            # 计算KS检验
            try:
                if dist_name == "lognormal":
                    ks_stat, ks_p = stats.kstest(data, 'lognorm', args=params)
                elif dist_name == "normal":
                    ks_stat, ks_p = stats.kstest(data, 'norm', args=params)
                elif dist_name == "exponential":
                    ks_stat, ks_p = stats.kstest(data, 'expon', args=params)
                elif dist_name == "inverse_gaussian":
                    # 逆高斯分布需要自定义KS检验
                    cdf_fitted = dist.cdf(np.sort(data), *params)
                    cdf_empirical = np.arange(1, len(data) + 1) / len(data)
                    ks_stat = np.max(np.abs(cdf_empirical - cdf_fitted))
                    ks_p = stats.kstwo.sf(ks_stat, len(data))
            except:
                ks_stat, ks_p = None, None

            results[dist_name] = {
                'params': params,
                'AIC': aic,
                'KS_p': ks_p
            }

            # 格式化输出参数
            if dist_name == "lognormal":
                param_str = f"shape={params[0]:.4f}, scale={params[2]:.4f}"
            elif dist_name == "normal":
                param_str = f"loc={params[0]:.4f}, scale={params[1]:.4f}"
            elif dist_name == "exponential":
                param_str = f"scale={params[1]:.4f}"
            elif dist_name == "inverse_gaussian":
                param_str = f"mu={params[0]:.4f}, scale={params[2]:.4f}"

            print(
                f"{dist_name:20} AIC: {aic:12.2f}  KS p值: {ks_p:.2e}" if ks_p else f"{dist_name:20} AIC: {aic:12.2f}  KS p值: N/A")
            print(f"                   参数: {param_str}")

        except Exception as e:
            print(f"{dist_name:20} 拟合失败: {e}")
            results[dist_name] = None

    # 确定最佳拟合（最小AIC）
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_fit = min(valid_results, key=lambda x: valid_results[x]['AIC'])
        best_aic = valid_results[best_fit]['AIC']
        print(f"\n最佳拟合分布: {best_fit} (AIC最小 = {best_aic:.2f})")

        # 计算ΔAIC（与最佳模型的差异）
        print(f"ΔAIC值:")
        for dist_name, result in valid_results.items():
            delta_aic = result['AIC'] - best_aic
            print(f"  {dist_name:20}: ΔAIC = {delta_aic:7.2f}")
    else:
        best_fit = None

    return best_fit, results


def analyze_long_tail(data):
    """
    判断数据是否呈现长尾分布

    标准:
    1. 偏度 > 1 (正偏/右偏)
    2. 峰度 > 3 (重尾)
    3. 前1%的数据占总值20%以上
    """
    data_abs = np.abs(data[data != 0])  # 取绝对值，排除零值

    # 计算统计量
    data_skew = skew(data_abs)
    data_kurtosis = kurtosis(data_abs, fisher=True)

    # 计算前1%占比
    sorted_abs = np.sort(data_abs)[::-1]
    top_1p = np.sum(sorted_abs[:int(len(sorted_abs) * 0.01)]) / np.sum(sorted_abs) * 100

    # 计算前5%和前10%占比
    top_5p = np.sum(sorted_abs[:int(len(sorted_abs) * 0.05)]) / np.sum(sorted_abs) * 100
    top_10p = np.sum(sorted_abs[:int(len(sorted_abs) * 0.10)]) / np.sum(sorted_abs) * 100

    print("\n长尾分布判断:")
    print("-" * 40)
    print(f"偏度: {data_skew:.3f} {'(>1, 正偏)' if data_skew > 1 else ''}")
    print(f"峰度: {data_kurtosis:.3f} {'(>3, 重尾)' if data_kurtosis > 3 else ''}")
    print(f"前1%连接强度占比: {top_1p:.1f}% {'(>20%, 枢纽连接)' if top_1p > 20 else ''}")
    print(f"前5%连接强度占比: {top_5p:.1f}%")
    print(f"前10%连接强度占比: {top_10p:.1f}%")

    # 计算百分位数
    percentiles = [50, 80, 90, 95, 99]
    print(f"\n连接强度百分位数:")
    for p in percentiles:
        p_value = np.percentile(data_abs, p)
        print(f"  {p:2d}%: {p_value:.6f}")

    # 判断
    is_long_tail = (data_skew > 1) or (data_kurtosis > 3) or (top_1p > 20)

    if is_long_tail:
        print("\n结论: ✅ 呈现长尾分布特征")
        print("      表明：大多数连接非常弱，少数连接极强")
        print("      支持'大脑网络由大量弱连接和少数关键的枢纽强连接构成'的假设")
    else:
        print("\n结论: ⚠ 未呈现典型长尾分布特征")

    return is_long_tail, {"skewness": data_skew, "kurtosis": data_kurtosis,
                          "top1p": top_1p, "top5p": top_5p, "top10p": top_10p,
                          "data_abs": data_abs}


def main():
    """主函数"""
    # 定义输出文件路径
    log_file_path = r"...\ECdata_distribution_plcb.txt"

    # 设置输出重定向
    original_stdout, log_file = setup_output_redirection(log_file_path)

    try:
        # 文件路径
        filepath = r"...\ECmatrix_plcb_reordered_withSub.npy"

        print("=" * 70)
        print("有效连接矩阵分布分析")
        print("=" * 70)
        print(f"分析时间: {np.datetime64('now', 's')}")
        print(f"日志文件: {log_file_path}")
        print(f"数据文件: {filepath}")

        # 1. 加载数据
        print(f"加载文件: {filepath}")
        try:
            ec_matrix = np.load(filepath)
            print(f"矩阵形状: {ec_matrix.shape}")

            # 检查矩阵是否为对称矩阵（可选）
            is_symmetric = np.allclose(ec_matrix, ec_matrix.T)
            print(f"矩阵对称性: {'对称' if is_symmetric else '非对称'}")
        except Exception as e:
            print(f"加载文件失败: {e}")
            return

        # 2. 提取非对角线元素
        n = ec_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        values = ec_matrix[mask].flatten()

        # 分离兴奋性和抑制性连接
        pos_values = values[values > 0]  # 兴奋性连接
        neg_values = np.abs(values[values < 0])  # 抑制性连接（取绝对值）
        zero_count = np.sum(values == 0)

        print(f"\n数据概况:")
        print(f"总连接数: {len(values):,}")
        print(f"兴奋性连接数: {len(pos_values):,} ({len(pos_values) / len(values) * 100:.1f}%)")
        print(f"抑制性连接数: {len(neg_values):,} ({len(neg_values) / len(values) * 100:.1f}%)")
        print(f"零值连接数: {zero_count:,} ({zero_count / len(values) * 100:.1f}%)")

        if len(pos_values) > 0:
            print(f"\n兴奋性连接强度统计:")
            print(f"  最小值: {np.min(pos_values):.6f}")
            print(f"  最大值: {np.max(pos_values):.6f}")
            print(f"  平均值: {np.mean(pos_values):.6f}")
            print(f"  中位数: {np.median(pos_values):.6f}")
            print(f"  标准差: {np.std(pos_values):.6f}")

        if len(neg_values) > 0:
            print(f"\n抑制性连接强度统计(绝对值):")
            print(f"  最小值: {np.min(neg_values):.6f}")
            print(f"  最大值: {np.max(neg_values):.6f}")
            print(f"  平均值: {np.mean(neg_values):.6f}")
            print(f"  中位数: {np.median(neg_values):.6f}")
            print(f"  标准差: {np.std(neg_values):.6f}")

        # 3. 长尾分布判断（所有连接取绝对值）
        print("\n" + "=" * 70)
        print("长尾分布分析 (所有连接绝对值)")
        print("=" * 70)
        is_long_tail, long_tail_stats = analyze_long_tail(values)

        # 4. 兴奋性连接分布拟合
        print("\n" + "=" * 70)
        print("兴奋性连接分布拟合分析")
        print("=" * 70)
        if len(pos_values) > 0:
            best_fit_pos, fit_results_pos = fit_distributions(pos_values, "兴奋性连接")
        else:
            print("无兴奋性连接数据")
            best_fit_pos, fit_results_pos = None, None

        # 5. 抑制性连接分布拟合
        print("\n" + "=" * 70)
        print("抑制性连接分布拟合分析")
        print("=" * 70)
        if len(neg_values) > 0:
            best_fit_neg, fit_results_neg = fit_distributions(neg_values, "抑制性连接")
        else:
            print("无抑制性连接数据")
            best_fit_neg, fit_results_neg = None, None

        # 6. 所有连接（绝对值）分布拟合（新增）
        print("\n" + "=" * 70)
        print("所有连接（绝对值）分布拟合分析")
        print("=" * 70)
        # 取所有非零连接的绝对值
        all_abs_values = long_tail_stats["data_abs"]

        if len(all_abs_values) > 0:
            best_fit_all, fit_results_all = fit_distributions(all_abs_values, "所有连接（绝对值）")
        else:
            print("无有效连接数据（绝对值）")
            best_fit_all, fit_results_all = None, None

        # 7. 总结
        print("\n" + "=" * 70)
        print("分析总结")
        print("=" * 70)
        print(f"1. 长尾分布判断: {'✅ 是' if is_long_tail else '⚠ 否'}")
        print(f"2. 兴奋性连接最佳拟合: {best_fit_pos if best_fit_pos else 'N/A'}")
        print(f"3. 抑制性连接最佳拟合: {best_fit_neg if best_fit_neg else 'N/A'}")
        print(f"4. 所有连接（绝对值）最佳拟合: {best_fit_all if best_fit_all else 'N/A'}")

        print("\n分析完成!")
        print(f"所有输出已保存到: {log_file_path}")

    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始stdout并关闭日志文件
        restore_output_redirection(original_stdout, log_file)
        print(f"分析日志已保存到: {log_file_path}")


if __name__ == "__main__":
    main()