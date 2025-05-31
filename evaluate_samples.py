import os
import time
import numpy as np
import librosa
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
import psutil
import gc
from tqdm import tqdm

# 设置随机种子，确保结果可重复
np.random.seed(42)
torch.manual_seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

def load_audio(file_path, sr=22050):
    """加载音频文件，返回波形和采样率"""
    try:
        wav, sr = librosa.load(file_path, sr=sr)
        return wav, sr
    except Exception as e:
        print(f"加载音频文件 {file_path} 时出错: {e}")
        return None, None

def extract_mfcc(wav, sr, n_mfcc=13):
    """提取MFCC特征"""
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_mel(wav, sr, n_mels=80, hop_length=256, win_length=1024):
    """提取梅尔频谱特征"""
    mel_spec = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, 
        hop_length=hop_length, win_length=win_length
    )
    # 转换到对数刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_f0(wav, sr, hop_length=256):
    """提取基频F0"""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        wav, fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )
    return f0

def align_audio_length(generated_wav, reference_wav):
    """将生成的音频长度与参考音频对齐，过长则截断，过短则补零"""
    if len(generated_wav) > len(reference_wav):
        # 截断
        return generated_wav[:len(reference_wav)]
    elif len(generated_wav) < len(reference_wav):
        # 补零
        padded = np.zeros_like(reference_wav)
        padded[:len(generated_wav)] = generated_wav
        return padded
    else:
        return generated_wav

def cosine_similarity(a, b):
    """计算两个向量之间的余弦相似度，结果范围在[-1, 1]之间，越接近1表示越相似"""
    # 将多维数组展平成一维
    a_flat = a.flatten()
    b_flat = b.flatten()
    # 计算余弦相似度
    return 1 - cosine(a_flat, b_flat)  # 余弦距离 = 1 - 余弦相似度，因此需要用1减去余弦距离

def compute_mcd(mfcc1, mfcc2):
    """计算Mel Cepstral Distortion (MCD)"""
    # 确保两个MFCC特征的时间维度相同
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]
    
    # 计算MCD
    diff = mfcc1 - mfcc2
    mcd = np.sqrt(np.sum(diff**2, axis=0))
    return np.mean(mcd)

def compute_f0_rmse(f0_generated, f0_reference):
    """计算F0的均方根误差 (RMSE)"""
    # 确保两个F0序列的长度相同
    min_len = min(len(f0_generated), len(f0_reference))
    f0_generated = f0_generated[:min_len]
    f0_reference = f0_reference[:min_len]
    
    # 只考虑两者都有有效F0值的帧
    mask = ~np.isnan(f0_generated) & ~np.isnan(f0_reference)
    if not np.any(mask):
        return np.nan  # 如果没有有效帧，返回NaN
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(
        f0_reference[mask], 
        f0_generated[mask]
    ))
    return rmse

def calculate_rtf(audio_length, processing_time):
    """计算实时率 (RTF)，RTF < 1 表示可以实时生成"""
    return processing_time / audio_length

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # 以MB为单位

def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # 以MB为单位
    else:
        return 0.0

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def evaluate_samples(demo_dir, f_dir, s_dir, output_dir='output/evaluation'):
    """评估生成的语音样本"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有真实样本的文件名
    ground_truth_files = [f for f in os.listdir(demo_dir) if f.endswith('.wav')]
    
    # 初始化结果存储
    results = {
        'f_model': {'cosine_sim': [], 'mcd': [], 'f0_rmse': [], 'rtf': [], 'memory': [], 'gpu_memory': []},
        's_model': {'cosine_sim': [], 'mcd': [], 'f0_rmse': [], 'rtf': [], 'memory': [], 'gpu_memory': []}
    }
    
    file_names = []
    
    # 处理每个样本
    for gt_file in tqdm(ground_truth_files, desc="评估语音样本"):
        base_name = gt_file.replace('_ground-truth.wav', '')
        file_names.append(base_name)
        
        # 加载真实样本
        gt_path = os.path.join(demo_dir, gt_file)
        gt_wav, sr = load_audio(gt_path)
        if gt_wav is None:
            print(f"跳过 {gt_file}，无法加载")
            continue
        
        gt_duration = len(gt_wav) / sr
        
        # 提取真实样本的特征
        gt_mel = extract_mel(gt_wav, sr)
        gt_mfcc = extract_mfcc(gt_wav, sr)
        gt_f0 = extract_f0(gt_wav, sr)
        
        # 评估 f 模型生成的样本
        f_file = f"{base_name}_f.wav"
        f_path = os.path.join(f_dir, f_file)
        if os.path.exists(f_path):
            # 测量处理时间
            start_time = time.time()
            mem_before = get_memory_usage()
            gpu_mem_before = get_gpu_memory_usage()
            
            f_wav, _ = load_audio(f_path, sr=sr)
            if f_wav is not None:
                # 对齐音频长度
                f_wav = align_audio_length(f_wav, gt_wav)
                
                # 提取特征
                f_mel = extract_mel(f_wav, sr)
                f_mfcc = extract_mfcc(f_wav, sr)
                f_f0 = extract_f0(f_wav, sr)
                
                # 计算指标
                cosine_sim = cosine_similarity(gt_mel, f_mel)
                mcd = compute_mcd(gt_mfcc, f_mfcc)
                f0_rmse = compute_f0_rmse(f_f0, gt_f0)
                
                # 计算处理时间和资源使用
                end_time = time.time()
                processing_time = end_time - start_time
                rtf = calculate_rtf(gt_duration, processing_time)
                mem_used = get_memory_usage() - mem_before
                gpu_mem_used = get_gpu_memory_usage() - gpu_mem_before
                
                # 存储结果
                results['f_model']['cosine_sim'].append(cosine_sim)
                results['f_model']['mcd'].append(mcd)
                results['f_model']['f0_rmse'].append(f0_rmse)
                results['f_model']['rtf'].append(rtf)
                results['f_model']['memory'].append(mem_used)
                results['f_model']['gpu_memory'].append(gpu_mem_used)
        
        # 清理内存
        clear_gpu_memory()
        
        # 评估 s 模型生成的样本
        s_file = f"{base_name}_s.wav"
        s_path = os.path.join(s_dir, s_file)
        if os.path.exists(s_path):
            # 测量处理时间
            start_time = time.time()
            mem_before = get_memory_usage()
            gpu_mem_before = get_gpu_memory_usage()
            
            s_wav, _ = load_audio(s_path, sr=sr)
            if s_wav is not None:
                # 对齐音频长度
                s_wav = align_audio_length(s_wav, gt_wav)
                
                # 提取特征
                s_mel = extract_mel(s_wav, sr)
                s_mfcc = extract_mfcc(s_wav, sr)
                s_f0 = extract_f0(s_wav, sr)
                
                # 计算指标
                cosine_sim = cosine_similarity(gt_mel, s_mel)
                mcd = compute_mcd(gt_mfcc, s_mfcc)
                f0_rmse = compute_f0_rmse(s_f0, gt_f0)
                
                # 计算处理时间和资源使用
                end_time = time.time()
                processing_time = end_time - start_time
                rtf = calculate_rtf(gt_duration, processing_time)
                mem_used = get_memory_usage() - mem_before
                gpu_mem_used = get_gpu_memory_usage() - gpu_mem_before
                
                # 存储结果
                results['s_model']['cosine_sim'].append(cosine_sim)
                results['s_model']['mcd'].append(mcd)
                results['s_model']['f0_rmse'].append(f0_rmse)
                results['s_model']['rtf'].append(rtf)
                results['s_model']['memory'].append(mem_used)
                results['s_model']['gpu_memory'].append(gpu_mem_used)
        
        # 清理内存
        clear_gpu_memory()
    
    # 计算平均值
    avg_results = {}
    for model in ['f_model', 's_model']:
        avg_results[model] = {}
        for metric in ['cosine_sim', 'mcd', 'f0_rmse', 'rtf', 'memory', 'gpu_memory']:
            values = results[model][metric]
            if values:
                avg_results[model][metric] = np.mean(values)
            else:
                avg_results[model][metric] = np.nan
    
    # 输出结果
    print("\n评估结果摘要:")
    print("-" * 80)
    print(f"{'指标':<15} {'新模型(f)':<15} {'原模型(s)':<15}")
    print("-" * 80)
    print(f"{'余弦相似度':<15} {avg_results['f_model']['cosine_sim']:<15.4f} {avg_results['s_model']['cosine_sim']:<15.4f}")
    print(f"{'MCD':<15} {avg_results['f_model']['mcd']:<15.4f} {avg_results['s_model']['mcd']:<15.4f}")
    print(f"{'F0 RMSE':<15} {avg_results['f_model']['f0_rmse']:<15.4f} {avg_results['s_model']['f0_rmse']:<15.4f}")
    print(f"{'RTF':<15} {avg_results['f_model']['rtf']:<15.4f} {avg_results['s_model']['rtf']:<15.4f}")
    print(f"{'内存使用(MB)':<15} {avg_results['f_model']['memory']:<15.4f} {avg_results['s_model']['memory']:<15.4f}")
    print(f"{'GPU内存(MB)':<15} {avg_results['f_model']['gpu_memory']:<15.4f} {avg_results['s_model']['gpu_memory']:<15.4f}")
    print("-" * 80)
    
    # 保存详细结果到文件
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write("语音合成评估详细结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("每个样本的评估结果:\n")
        f.write("-" * 80 + "\n")
        for i, file_name in enumerate(file_names):
            f.write(f"样本: {file_name}\n")
            if i < len(results['f_model']['cosine_sim']):
                f.write(f"  新模型(f):\n")
                f.write(f"    余弦相似度: {results['f_model']['cosine_sim'][i]:.4f}\n")
                f.write(f"    MCD: {results['f_model']['mcd'][i]:.4f}\n")
                f.write(f"    F0 RMSE: {results['f_model']['f0_rmse'][i]:.4f}\n")
                f.write(f"    RTF: {results['f_model']['rtf'][i]:.4f}\n")
                f.write(f"    内存使用(MB): {results['f_model']['memory'][i]:.4f}\n")
                f.write(f"    GPU内存(MB): {results['f_model']['gpu_memory'][i]:.4f}\n")
            
            if i < len(results['s_model']['cosine_sim']):
                f.write(f"  原模型(s):\n")
                f.write(f"    余弦相似度: {results['s_model']['cosine_sim'][i]:.4f}\n")
                f.write(f"    MCD: {results['s_model']['mcd'][i]:.4f}\n")
                f.write(f"    F0 RMSE: {results['s_model']['f0_rmse'][i]:.4f}\n")
                f.write(f"    RTF: {results['s_model']['rtf'][i]:.4f}\n")
                f.write(f"    内存使用(MB): {results['s_model']['memory'][i]:.4f}\n")
                f.write(f"    GPU内存(MB): {results['s_model']['gpu_memory'][i]:.4f}\n")
            f.write("\n")
        
        f.write("\n平均结果:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<15} {'新模型(f)':<15} {'原模型(s)':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'余弦相似度':<15} {avg_results['f_model']['cosine_sim']:<15.4f} {avg_results['s_model']['cosine_sim']:<15.4f}\n")
        f.write(f"{'MCD':<15} {avg_results['f_model']['mcd']:<15.4f} {avg_results['s_model']['mcd']:<15.4f}\n")
        f.write(f"{'F0 RMSE':<15} {avg_results['f_model']['f0_rmse']:<15.4f} {avg_results['s_model']['f0_rmse']:<15.4f}\n")
        f.write(f"{'RTF':<15} {avg_results['f_model']['rtf']:<15.4f} {avg_results['s_model']['rtf']:<15.4f}\n")
        f.write(f"{'内存使用(MB)':<15} {avg_results['f_model']['memory']:<15.4f} {avg_results['s_model']['memory']:<15.4f}\n")
        f.write(f"{'GPU内存(MB)':<15} {avg_results['f_model']['gpu_memory']:<15.4f} {avg_results['s_model']['gpu_memory']:<15.4f}\n")
    
    # 绘制结果对比图表
    plot_comparison_charts(results, avg_results, output_dir)
    
    return avg_results

def plot_comparison_charts(results, avg_results, output_dir):
    """绘制模型评估结果的对比图表"""
    # 1. 余弦相似度对比图
    plt.figure(figsize=(10, 6))
    plt.bar(['新模型(f)', '原模型(s)'], 
            [avg_results['f_model']['cosine_sim'], avg_results['s_model']['cosine_sim']])
    plt.title('模型生成样本与真实样本的余弦相似度对比')
    plt.ylabel('余弦相似度 (越高越好)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, '余弦相似度对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MCD对比图
    plt.figure(figsize=(10, 6))
    plt.bar(['新模型(f)', '原模型(s)'], 
            [avg_results['f_model']['mcd'], avg_results['s_model']['mcd']])
    plt.title('模型生成样本的Mel Cepstral Distortion对比')
    plt.ylabel('MCD (越低越好)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'MCD对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F0 RMSE对比图
    plt.figure(figsize=(10, 6))
    plt.bar(['新模型(f)', '原模型(s)'], 
            [avg_results['f_model']['f0_rmse'], avg_results['s_model']['f0_rmse']])
    plt.title('模型生成样本的基频(F0)均方根误差对比')
    plt.ylabel('F0 RMSE (越低越好)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'F0_RMSE对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. RTF对比图
    plt.figure(figsize=(10, 6))
    plt.bar(['新模型(f)', '原模型(s)'], 
            [avg_results['f_model']['rtf'], avg_results['s_model']['rtf']])
    plt.title('模型的实时率(RTF)对比')
    plt.ylabel('RTF (越低越好，<1表示实时)')
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.5, label='实时阈值')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'RTF对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 内存使用对比图
    plt.figure(figsize=(10, 6))
    plt.bar(['新模型(f)', '原模型(s)'], 
            [avg_results['f_model']['memory'], avg_results['s_model']['memory']])
    plt.title('模型的内存使用对比')
    plt.ylabel('内存使用 (MB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, '内存使用对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. GPU内存使用对比图
    if torch.cuda.is_available():
        plt.figure(figsize=(10, 6))
        plt.bar(['新模型(f)', '原模型(s)'], 
                [avg_results['f_model']['gpu_memory'], avg_results['s_model']['gpu_memory']])
        plt.title('模型的GPU内存使用对比')
        plt.ylabel('GPU内存使用 (MB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'GPU内存使用对比.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. 综合评分雷达图
    metrics = ['余弦相似度', 'MCD (反转)', 'F0 RMSE (反转)', 'RTF (反转)', '内存 (反转)']
    
    # 对于MCD、F0 RMSE、RTF和内存，值越低越好，所以需要反转
    f_values = [
        avg_results['f_model']['cosine_sim'],
        1 / (avg_results['f_model']['mcd'] + 1e-10),  # 加一个小值避免除零
        1 / (avg_results['f_model']['f0_rmse'] + 1e-10),
        1 / (avg_results['f_model']['rtf'] + 1e-10),
        1 / (avg_results['f_model']['memory'] + 1e-10)
    ]
    
    s_values = [
        avg_results['s_model']['cosine_sim'],
        1 / (avg_results['s_model']['mcd'] + 1e-10),
        1 / (avg_results['s_model']['f0_rmse'] + 1e-10),
        1 / (avg_results['s_model']['rtf'] + 1e-10),
        1 / (avg_results['s_model']['memory'] + 1e-10)
    ]
    
    # 归一化
    max_values = np.maximum(f_values, s_values)
    f_values = f_values / max_values
    s_values = s_values / max_values
    
    # 绘制雷达图
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # 闭合多边形 - 确保维度匹配
    f_values = np.append(f_values, f_values[0])
    s_values = np.append(s_values, s_values[0])
    angles = np.append(angles, angles[0])
    metrics.append(metrics[0])
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, f_values, 'o-', linewidth=2, label='新模型(f)')
    ax.plot(angles, s_values, 'o-', linewidth=2, label='原模型(s)')
    ax.fill(angles, f_values, alpha=0.25)
    ax.fill(angles, s_values, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('模型性能综合评价雷达图', y=1.08)
    plt.savefig(os.path.join(output_dir, '综合评分雷达图.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    demo_dir = "demo/demo"  # 真实样本目录
    f_dir = "demo/f"        # 新模型样本目录
    s_dir = "demo/s"        # 原模型样本目录
    
    # 创建输出目录
    output_dir = "output/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    print("开始评估语音样本...")
    avg_results = evaluate_samples(demo_dir, f_dir, s_dir, output_dir)
    
    print(f"\n所有评估结果和图表已保存到 {output_dir} 目录") 