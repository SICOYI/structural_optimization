import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

# 设置 seaborn 样式
sns.set_theme(style="whitegrid")

# 读取JSON文件
with open('strain_energy_optimization_timings.json', 'r') as f:
    data = json.load(f)

# 提取三种时间数据
assembly_times = np.array(data['assembly_times'])
fe_times = np.array(data['fe_times'])
backward_times = np.array(data['backward_times'])


# 计算统计量
def calculate_stats(times, name):
    return {
        'name': name,
        'mean': np.mean(times),
        'p95': np.percentile(times, 95),
        'p05': np.percentile(times, 5)
    }


stats = [
    calculate_stats(assembly_times, 'Assembly Times'),
    calculate_stats(fe_times, 'FE Times'),
    calculate_stats(backward_times, 'Backward Times')
]

# 打印统计结果
print("\nTime Statistics:")
print("=" * 50)
for stat in stats:
    print(f"{stat['name']}:")
    print(f"  Mean: {stat['mean']:.6f} s")
    print(f"  95th percentile: {stat['p95']:.6f} s")
    print(f"  5th percentile: {stat['p05']:.6f} s")
    print("-" * 50)

# 设置颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 1. 绘制直方图和PDF图（标记关键分位数）
plt.figure(figsize=(18, 5))

for i, (times, color, stat) in enumerate(zip(
        [assembly_times, fe_times, backward_times],
        colors,
        stats
), 1):
    plt.subplot(1, 3, i)
    ax = sns.histplot(times, bins=20, color=color, edgecolor='white', alpha=0.7, kde=True, stat="density")

    # 标记关键分位数
    plt.axvline(stat['mean'], color='red', linestyle='--', label=f"Mean: {stat['mean']:.4f}s")
    plt.axvline(stat['p95'], color='green', linestyle=':', label=f"95%: {stat['p95']:.4f}s")
    plt.axvline(stat['p05'], color='blue', linestyle=':', label=f"5%: {stat['p05']:.4f}s")

    plt.title(f'{stat["name"]} Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()

# 2. 绘制单独的PDF曲线比较图
plt.figure(figsize=(10, 6))

# 计算KDE
kde_assembly = gaussian_kde(assembly_times)
kde_fe = gaussian_kde(fe_times)
kde_backward = gaussian_kde(backward_times)

# 创建x轴范围
x_assembly = np.linspace(min(assembly_times), max(assembly_times), 100)
x_fe = np.linspace(min(fe_times), max(fe_times), 100)
x_backward = np.linspace(min(backward_times), max(backward_times), 100)

# 绘制PDF曲线
plt.plot(x_assembly, kde_assembly(x_assembly), color=colors[0], label='Assembly Times', linewidth=2)
plt.plot(x_fe, kde_fe(x_fe), color=colors[1], label='FE Times', linewidth=2)
plt.plot(x_backward, kde_backward(x_backward), color=colors[2], label='Backward Times', linewidth=2)

plt.title('Probability Density Functions Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 3. 绘制箱线图（保留原代码）
plt.figure(figsize=(10, 6))
plt.boxplot([assembly_times, fe_times, backward_times],
            labels=['Assembly Times', 'FE Times', 'Backward Times'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='darkblue'),
            medianprops=dict(color='red'))
plt.title('Comparison of Time Distributions')
plt.ylabel('Time (s)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()