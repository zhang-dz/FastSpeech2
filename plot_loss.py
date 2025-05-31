import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统黑体字体路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 读取老模型的损失数据
old_losses = []
with open('output/log/validation_log.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if '总损失:' in line:
            loss = float(line.split('总损失:')[1].split(',')[0].strip())
            old_losses.append(loss)

# 读取新模型的损失数据
new_losses = []
with open('output/log/attention_val/log.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if 'Total Loss:' in line:
            loss = float(line.split('Total Loss:')[1].split(',')[0].strip())
            new_losses.append(loss)

# 创建步长为100的x轴数据，限制在500k以内
x = np.arange(0, 500000, 100)

# 对数据进行插值，确保数据点数量匹配
from scipy.interpolate import interp1d

# 创建原始数据的x轴（每1000步一个点）
x_old = np.arange(0, len(old_losses) * 1000, 1000)
x_new = np.arange(0, len(new_losses) * 1000, 1000)

# 创建插值函数
f_old = interp1d(x_old, old_losses, kind='linear', bounds_error=False, fill_value=(old_losses[0], old_losses[-1]))
f_new = interp1d(x_new, new_losses, kind='linear', bounds_error=False, fill_value=(new_losses[0], new_losses[-1]))

# 使用插值函数生成新的数据点
y_old = f_old(x)
y_new = f_new(x)

# 绘制对比图
plt.figure(figsize=(15, 8))
plt.plot(x, y_old, label='老模型', color='blue')
plt.plot(x, y_new, label='新模型', color='red')

plt.xlabel('迭代次数', fontproperties=font_prop, fontsize=12)
plt.ylabel('验证损失', fontproperties=font_prop, fontsize=12)
plt.title('模型验证损失对比 (前500k迭代)', fontproperties=font_prop, fontsize=14)
plt.legend(prop=font_prop, fontsize=12)
plt.grid(True)

# 调整布局以确保标签完全显示
plt.tight_layout()

# 保存图片
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 