import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .computeShan import compute_mel_spectral_entropy
import librosa

class CurriculumDataset(Dataset):
    def __init__(
        self,
        filename,
        preprocess_config,
        train_config,
        sort=False,
        drop_last=False,
        num_levels=5,  # 难度等级数量
        current_level=1,  # 当前训练阶段
        entropy_threshold=None,  # 熵值阈值
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.num_levels = num_levels
        self.current_level = current_level
        self.entropy_threshold = entropy_threshold

        # 加载基本数据
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        
        # 计算每个样本的熵值
        self.entropies = self._compute_entropies()
        
        # 根据熵值对样本进行排序和分层
        self.sample_indices = self._stratify_samples()
        
        self.sort = sort
        self.drop_last = drop_last

    def _compute_entropies(self):
        """计算所有样本的Mel-谱图信息熵"""
        entropies = []
        print("计算样本熵值...")
        for i in tqdm(range(len(self.basename))):
            basename = self.basename[i]
            speaker = self.speaker[i]
            mel_path = os.path.join(
                self.preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            mel = np.load(mel_path)
            
            # 将mel谱图转换为波形
            wav = librosa.feature.inverse.mel_to_audio(
                mel,
                sr=22050,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
            )
            
            # 计算熵值
            _, H_avg, _, _ = compute_mel_spectral_entropy(
                wav,
                sr=22050,
                n_mels=80,
                hop_length=256,
                win_length=1024
            )
            entropies.append(H_avg)
        
        return np.array(entropies)

    def _stratify_samples(self):
        """根据熵值对样本进行分层"""
        # 对样本按熵值排序
        sorted_indices = np.argsort(self.entropies)
        
        # 计算每层的样本数量
        samples_per_level = len(sorted_indices) // self.num_levels
        
        # 分层
        stratified_indices = []
        for i in range(self.num_levels):
            start_idx = i * samples_per_level
            end_idx = (i + 1) * samples_per_level if i < self.num_levels - 1 else len(sorted_indices)
            stratified_indices.append(sorted_indices[start_idx:end_idx])
        
        return stratified_indices

    def __len__(self):
        """返回当前难度等级可用的样本数量"""
        return len(self.sample_indices[self.current_level - 1])

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        # 获取当前难度等级的实际索引
        actual_idx = self.sample_indices[self.current_level - 1][idx]
        
        basename = self.basename[actual_idx]
        speaker = self.speaker[actual_idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[actual_idx]
        phone = np.array(text_to_sequence(self.text[actual_idx], self.cleaners))
        
        # 加载特征
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "entropy": self.entropies[actual_idx],
        }

        return sample

    def process_meta(self, filename):
        """处理元数据文件"""
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        """重新处理数据批次"""
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        entropies = [data[idx]["entropy"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        entropies = np.array(entropies)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            entropies,
        )

    def collate_fn(self, data):
        """数据批处理函数"""
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

    def advance_level(self, val_loss=None):
        """进入下一个难度等级"""
        if self.current_level < self.num_levels:
            self.current_level += 1
            return True
        return False

    def get_level_stats(self):
        """获取当前难度等级的统计信息"""
        current_indices = self.sample_indices[self.current_level - 1]
        current_entropies = self.entropies[current_indices]
        return {
            "level": self.current_level,
            "num_samples": len(current_indices),
            "mean_entropy": np.mean(current_entropies),
            "std_entropy": np.std(current_entropies),
            "min_entropy": np.min(current_entropies),
            "max_entropy": np.max(current_entropies),
        } 