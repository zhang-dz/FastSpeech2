import matplotlib.pyplot as plt
import numpy as np
import re
import os

# 创建保存图的目录
os.makedirs('output/plots', exist_ok=True)

# 读取日志文件函数
def read_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# 解析训练日志数据
def parse_train_log(content, is_attention_model=False):
    pattern = r'Step (\d+)/\d+, Total Loss: ([\d\.]+), Mel Loss: ([\d\.]+), Mel PostNet Loss: ([\d\.]+), Pitch Loss: ([\d\.]+), Energy Loss: ([\d\.]+), Duration Loss: ([\d\.]+)'
    matches = re.findall(pattern, content)
    
    steps = []
    total_losses = []
    mel_losses = []
    mel_postnet_losses = []
    pitch_losses = []
    energy_losses = []
    duration_losses = []
    
    for match in matches:
        steps.append(int(match[0]))
        total_losses.append(float(match[1]))
        mel_losses.append(float(match[2]))
        mel_postnet_losses.append(float(match[3]))
        pitch_losses.append(float(match[4]))
        energy_losses.append(float(match[5]))
        duration_losses.append(float(match[6]))
    
    return {
        'steps': steps,
        'total_losses': total_losses,
        'mel_losses': mel_losses,
        'mel_postnet_losses': mel_postnet_losses,
        'pitch_losses': pitch_losses,
        'energy_losses': energy_losses,
        'duration_losses': duration_losses
    }

# 解析验证日志数据
def parse_validation_log(content, is_attention_model=False):
    if is_attention_model:
        pattern = r'Validation Step (\d+), Total Loss: ([\d\.]+), Mel Loss: ([\d\.]+), Mel PostNet Loss: ([\d\.]+), Pitch Loss: ([\d\.]+), Energy Loss: ([\d\.]+), Duration Loss: ([\d\.]+)'
    else:
        pattern = r'验证步骤 (\d+), 总损失: ([\d\.]+), Mel损失: ([\d\.]+), Mel PostNet损失: ([\d\.]+), 音高损失: ([\d\.]+), 能量损失: ([\d\.]+), 持续时间损失: ([\d\.]+)'
    
    matches = re.findall(pattern, content)
    
    steps = []
    total_losses = []
    mel_losses = []
    mel_postnet_losses = []
    pitch_losses = []
    energy_losses = []
    duration_losses = []
    
    for match in matches:
        steps.append(int(match[0]))
        total_losses.append(float(match[1]))
        mel_losses.append(float(match[2]))
        mel_postnet_losses.append(float(match[3]))
        pitch_losses.append(float(match[4]))
        energy_losses.append(float(match[5]))
        duration_losses.append(float(match[6]))
    
    return {
        'steps': steps,
        'total_losses': total_losses,
        'mel_losses': mel_losses,
        'mel_postnet_losses': mel_postnet_losses,
        'pitch_losses': pitch_losses,
        'energy_losses': energy_losses,
        'duration_losses': duration_losses
    }

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 读取日志文件
attn_train_content = read_log_file('output/log/attention_train/log.txt')
attn_val_content = read_log_file('output/log/attention_val/log.txt')
new_train_content = read_log_file('output/log/train_log.txt')
new_val_content = read_log_file('output/log/validation_log.txt')

# 解析日志数据
attn_train_data = parse_train_log(attn_train_content, True)
attn_val_data = parse_validation_log(attn_val_content, True)
new_train_data = parse_train_log(new_train_content)
new_val_data = parse_validation_log(new_val_content)

# 1. 绘制总损失训练趋势对比图
plt.figure(figsize=(12, 6))
plt.plot(attn_train_data['steps'], attn_train_data['total_losses'], label='原模型训练损失')
plt.plot(new_train_data['steps'], new_train_data['total_losses'], label='新模型训练损失')
plt.title('两个模型训练总损失对比')
plt.xlabel('训练步数')
plt.ylabel('总损失')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/总损失训练对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 绘制验证损失对比图
plt.figure(figsize=(12, 6))
plt.plot(attn_val_data['steps'], attn_val_data['total_losses'], label='原模型验证损失')
plt.plot(new_val_data['steps'], new_val_data['total_losses'], label='新模型验证损失')
plt.title('两个模型验证总损失对比')
plt.xlabel('验证步数')
plt.ylabel('总损失')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/总损失验证对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 绘制原模型各损失训练曲线
plt.figure(figsize=(14, 8))
plt.plot(attn_train_data['steps'], attn_train_data['total_losses'], label='总损失')
plt.plot(attn_train_data['steps'], attn_train_data['mel_losses'], label='Mel损失')
plt.plot(attn_train_data['steps'], attn_train_data['mel_postnet_losses'], label='Mel PostNet损失')
plt.plot(attn_train_data['steps'], attn_train_data['pitch_losses'], label='音高损失')
plt.plot(attn_train_data['steps'], attn_train_data['energy_losses'], label='能量损失')
plt.plot(attn_train_data['steps'], attn_train_data['duration_losses'], label='持续时间损失')
plt.title('原模型训练损失曲线')
plt.xlabel('训练步数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/原模型训练损失曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 绘制新模型各损失训练曲线
plt.figure(figsize=(14, 8))
plt.plot(new_train_data['steps'], new_train_data['total_losses'], label='总损失')
plt.plot(new_train_data['steps'], new_train_data['mel_losses'], label='Mel损失')
plt.plot(new_train_data['steps'], new_train_data['mel_postnet_losses'], label='Mel PostNet损失')
plt.plot(new_train_data['steps'], new_train_data['pitch_losses'], label='音高损失')
plt.plot(new_train_data['steps'], new_train_data['energy_losses'], label='能量损失')
plt.plot(new_train_data['steps'], new_train_data['duration_losses'], label='持续时间损失')
plt.title('新模型训练损失曲线')
plt.xlabel('训练步数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/新模型训练损失曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 各组件损失对比 - Mel损失
plt.figure(figsize=(12, 6))
plt.plot(attn_train_data['steps'], attn_train_data['mel_losses'], label='原模型Mel损失')
plt.plot(new_train_data['steps'], new_train_data['mel_losses'], label='新模型Mel损失')
plt.title('两个模型Mel损失对比')
plt.xlabel('训练步数')
plt.ylabel('Mel损失')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/Mel损失对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 各组件损失对比 - 音高损失
plt.figure(figsize=(12, 6))
plt.plot(attn_train_data['steps'], attn_train_data['pitch_losses'], label='原模型音高损失')
plt.plot(new_train_data['steps'], new_train_data['pitch_losses'], label='新模型音高损失')
plt.title('两个模型音高损失对比')
plt.xlabel('训练步数')
plt.ylabel('音高损失')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/音高损失对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 各组件损失对比 - 能量损失
plt.figure(figsize=(12, 6))
plt.plot(attn_train_data['steps'], attn_train_data['energy_losses'], label='原模型能量损失')
plt.plot(new_train_data['steps'], new_train_data['energy_losses'], label='新模型能量损失')
plt.title('两个模型能量损失对比')
plt.xlabel('训练步数')
plt.ylabel('能量损失')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/能量损失对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. 各组件损失对比 - 持续时间损失
plt.figure(figsize=(12, 6))
plt.plot(attn_train_data['steps'], attn_train_data['duration_losses'], label='原模型持续时间损失')
plt.plot(new_train_data['steps'], new_train_data['duration_losses'], label='新模型持续时间损失')
plt.title('两个模型持续时间损失对比')
plt.xlabel('训练步数')
plt.ylabel('持续时间损失')
plt.legend()
plt.grid(True)
plt.savefig('output/plots/持续时间损失对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. 验证集上的对比 - 创建直方图比较最后十次验证的平均损失
def get_last_n_avg(data, n=10):
    if len(data) < n:
        n = len(data)
    return np.mean(data[-n:])

attn_last_total = get_last_n_avg(attn_val_data['total_losses'])
attn_last_mel = get_last_n_avg(attn_val_data['mel_losses'])
attn_last_pitch = get_last_n_avg(attn_val_data['pitch_losses'])
attn_last_energy = get_last_n_avg(attn_val_data['energy_losses'])
attn_last_duration = get_last_n_avg(attn_val_data['duration_losses'])

new_last_total = get_last_n_avg(new_val_data['total_losses'])
new_last_mel = get_last_n_avg(new_val_data['mel_losses'])
new_last_pitch = get_last_n_avg(new_val_data['pitch_losses'])
new_last_energy = get_last_n_avg(new_val_data['energy_losses'])
new_last_duration = get_last_n_avg(new_val_data['duration_losses'])

# 最后验证损失对比直方图
labels = ['总损失', 'Mel损失', '音高损失', '能量损失', '持续时间损失']
attn_values = [attn_last_total, attn_last_mel, attn_last_pitch, attn_last_energy, attn_last_duration]
new_values = [new_last_total, new_last_mel, new_last_pitch, new_last_energy, new_last_duration]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, attn_values, width, label='原模型')
rects2 = ax.bar(x + width/2, new_values, width, label='新模型')

ax.set_ylabel('损失值')
ax.set_title('两个模型最终验证损失对比')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('output/plots/最终验证损失对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. 训练收敛速度对比 - 找到模型达到特定损失阈值的步数
def find_convergence_step(losses, threshold):
    for i, loss in enumerate(losses):
        if loss <= threshold:
            return i
    return len(losses) - 1  # 如果没有达到阈值，返回最后一步

thresholds = [3.0, 2.5, 2.0, 1.8]
attn_convergence_steps = []
new_convergence_steps = []

for threshold in thresholds:
    attn_step = find_convergence_step(attn_train_data['total_losses'], threshold)
    new_step = find_convergence_step(new_train_data['total_losses'], threshold)
    
    if attn_step < len(attn_train_data['steps']):
        attn_steps = attn_train_data['steps'][attn_step]
    else:
        attn_steps = None
        
    if new_step < len(new_train_data['steps']):
        new_steps = new_train_data['steps'][new_step]
    else:
        new_steps = None
    
    attn_convergence_steps.append(attn_steps)
    new_convergence_steps.append(new_steps)

# 绘制收敛速度对比图
valid_thresholds = []
valid_attn_steps = []
valid_new_steps = []

for i, threshold in enumerate(thresholds):
    if attn_convergence_steps[i] is not None and new_convergence_steps[i] is not None:
        valid_thresholds.append(threshold)
        valid_attn_steps.append(attn_convergence_steps[i])
        valid_new_steps.append(new_convergence_steps[i])

if valid_thresholds:
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(valid_thresholds))
    
    plt.bar(x - width/2, valid_attn_steps, width, label='原模型')
    plt.bar(x + width/2, valid_new_steps, width, label='新模型')
    
    plt.title('两个模型达到特定损失阈值所需的训练步数')
    plt.xlabel('损失阈值')
    plt.ylabel('训练步数')
    plt.xticks(x, valid_thresholds)
    
    # 添加数值标签
    for i, v in enumerate(valid_attn_steps):
        plt.text(i - width/2, v + 100, str(v), ha='center')
    
    for i, v in enumerate(valid_new_steps):
        plt.text(i + width/2, v + 100, str(v), ha='center')
    
    plt.legend()
    plt.grid(True)
    plt.savefig('output/plots/收敛速度对比.png', dpi=300, bbox_inches='tight')
    plt.close()

print("所有图表已保存到 output/plots 目录") 