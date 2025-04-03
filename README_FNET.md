# FastSpeech2 with FNet

本项目实现了一个基于FNet的FastSpeech2变体，使用傅里叶变换替代了原始FastSpeech2中的自注意力机制。

## 简介

FastSpeech2是一个非自回归的文本到语音合成模型，它使用Transformer架构来处理文本和音频特征。在原始的FastSpeech2中，编码器和解码器都使用了基于自注意力机制的Transformer块。

在这个变体中，我们用FNet（Google Research提出的一种使用傅里叶变换的模型）替代了自注意力机制。FNet的主要优势包括：

1. **计算效率高**：FNet的计算复杂度为O(n log n)，比自注意力的O(n²)更高效
2. **训练速度快**：FNet在GPU上训练速度比传统Transformer快得多
3. **内存占用小**：傅里叶变换层没有可学习参数，因此模型大小更小
4. **性能损失有限**：在许多任务上，FNet可以达到Transformer模型的92-97%的性能

## 文件结构

- `transformer/FourierLayers.py`: 实现了FourierTransform和FourierBlock
- `transformer/FourierModels.py`: 实现了FourierEncoder和FourierDecoder
- `model/fastspeech2_fnet.py`: 实现了FastSpeech2FNet模型
- `example_fnet.py`: 示例脚本，展示如何使用FastSpeech2FNet模型
- `synthesize_fnet.py`: 合成脚本，用于使用FastSpeech2FNet模型合成语音
- `train_fnet.py`: 训练脚本，用于训练FastSpeech2FNet模型

## 使用方法

### 安装依赖

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 测试模型

要测试FastSpeech2FNet模型的结构和参数，可以运行：

```bash
python example_fnet.py --preprocess_config config/LJSpeech/preprocess.yaml --model_config config/LJSpeech/model.yaml
```

### 训练模型

要从头开始训练FastSpeech2FNet模型，可以运行：

```bash
python train_fnet.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

如果您有预训练的FastSpeech2模型，可以从该模型继续训练：

```bash
python train_fnet.py --restore_step 900000 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

训练过程中的模型检查点将保存在`output/ckpt/LJSpeech/fnet`目录中。

### 合成语音

训练完成后，您可以使用训练好的模型合成语音：

```bash
python synthesize_fnet.py --restore_step 100000 --text "This is a test sentence for FastSpeech2 with FNet." --preprocess_config config/LJSpeech/preprocess.yaml --model_config config/LJSpeech/model.yaml --train_config config/LJSpeech/train.yaml
```

合成的语音将保存在`output/result/LJSpeech/fnet`目录中。

您还可以通过调整以下参数来控制合成语音的特性：
- `--pitch_control`: 控制音高（默认为1.0）
- `--energy_control`: 控制能量/音量（默认为1.0）
- `--duration_control`: 控制语速（默认为1.0）

例如，要生成音高提高20%、语速降低10%的语音：

```bash
python synthesize_fnet.py --restore_step 100000 --text "This is a test sentence for FastSpeech2 with FNet." --preprocess_config config/LJSpeech/preprocess.yaml --model_config config/LJSpeech/model.yaml --train_config config/LJSpeech/train.yaml --pitch_control 1.2 --duration_control 1.1
```

## 性能比较

理论上，FastSpeech2FNet应该具有以下特点：

1. **训练速度更快**：由于使用了傅里叶变换替代自注意力，训练速度应该显著提高
2. **内存占用更小**：模型参数更少，内存占用更小
3. **推理速度更快**：推理时计算复杂度更低
4. **可能的性能损失**：在某些情况下，可能会有轻微的性能损失

## 注意事项

1. 这个实现需要PyTorch 1.7+，因为它使用了`torch.fft.fftn`函数
2. 傅里叶变换是一种线性变换，可能在捕获某些复杂模式时不如自注意力
3. 您可能需要调整模型超参数以获得最佳性能

## 参考

- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
- [Google Research FNet](https://github.com/google-research/google-research/tree/master/f_net) 