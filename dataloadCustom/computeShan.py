import numpy as np
import librosa

def compute_mel_spectral_entropy(y, sr, 
                                  n_mels=80, 
                                  hop_length=256, 
                                  win_length=1024, 
                                  power=2.0, 
                                  log_base=np.e):
    """
    计算语音样本的Mel-谱图信息熵

    参数:
        y           : 1D numpy array，语音波形
        sr          : int，采样率
        n_mels      : int，Mel通道数（默认80）
        hop_length  : int，帧移
        win_length  : int，窗长
        power       : float，功率谱指数（默认为2，表示能量）
        log_base    : 对数底（np.e 表示自然对数；2 表示bit单位）

    返回:
        H_t         : 1D numpy array，每帧的熵值
        H_avg       : float，所有帧的平均熵
        H_median    : float，熵的中位数
        H_std       : float，熵的标准差
    """
    # 1. 计算Mel谱图 (功率谱)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, 
        win_length=win_length, power=power
    )  # shape: (n_mels, T)

    # 2. 帧归一化为概率分布
    eps = 1e-10  # 防止log(0)
    mel_spec += eps
    mel_spec /= np.sum(mel_spec, axis=0, keepdims=True)  # 按列归一化，每一帧为概率分布

    # 3. 熵计算
    if log_base == 2:
        log_fn = lambda x: np.log2(x)
    elif log_base == np.e:
        log_fn = lambda x: np.log(x)
    else:
        log_fn = lambda x: np.log(x) / np.log(log_base)

    H_t = -np.sum(mel_spec * log_fn(mel_spec), axis=0)  # shape: (T,)

    # 4. 全局统计量
    H_avg = np.mean(H_t)
    H_median = np.median(H_t)
    H_std = np.std(H_t)

    return H_t, H_avg, H_median, H_std
