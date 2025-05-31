import torch
from model.fastspeech2 import FastSpeech2
from model.fastspeech2_fnet import FastSpeech2FNet
import json
import yaml
import os

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_module_parameters(module, indent=0):
    """递归打印模块及其子模块的参数量"""
    total_params = count_parameters(module)
    indent_str = "  " * indent
    print(f"{indent_str}{module.__class__.__name__}: {total_params:,}")
    
    # 递归打印子模块
    for name, child in module.named_children():
        if child is not None:  # 确保模块不是None
            print(f"{indent_str}  {name}: {count_parameters(child):,}")

def analyze_model(model, model_name):
    print(f"\n{'-'*20} {model_name} 分析 {'-'*20}")
    
    # 移除PostNet
    model.postnet = None
    
    # 计算总参数量
    total_params = count_parameters(model)
    print(f"\n模型总参数量（不含PostNet）: {total_params:,}")
    
    # 详细打印各组件参数量
    print("\n详细参数量统计:")
    print("\n1. Encoder详细信息:")
    print_module_parameters(model.encoder, indent=1)
    
    print("\n2. Variance Adaptor详细信息:")
    print_module_parameters(model.variance_adaptor, indent=1)
    
    print("\n3. Decoder详细信息:")
    print_module_parameters(model.decoder, indent=1)
    
    print("\n4. Mel Linear层:")
    print(f"  参数量: {count_parameters(model.mel_linear):,}")
    
    if model.speaker_emb is not None:
        print("\n5. Speaker Embedding:")
        print(f"  参数量: {count_parameters(model.speaker_emb):,}")
    
    # 验证编码器和解码器结构
    print("\n编码器和解码器参数对比:")
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    print(f"Encoder总参数量: {encoder_params:,}")
    print(f"Decoder总参数量: {decoder_params:,}")
    if encoder_params != decoder_params:
        print("警告：编码器和解码器参数量不一致！")
    
def main():
    try:
        # 加载配置文件
        config_dir = "config/LJSpeech"
        preprocess_path = os.path.join(config_dir, "preprocess.yaml")
        model_path = os.path.join(config_dir, "model.yaml")
        
        print(f"正在加载配置文件...")
        print(f"Preprocess config: {preprocess_path}")
        print(f"Model config: {model_path}")
        
        with open(preprocess_path, "r", encoding='utf-8') as f:
            preprocess_config = yaml.safe_load(f)
        with open(model_path, "r", encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        
        print("配置文件加载成功！")
        
        # 创建并分析原始FastSpeech2模型
        print("\n创建FastSpeech2模型...")
        fs2_model = FastSpeech2(preprocess_config, model_config)
        analyze_model(fs2_model, "FastSpeech2")
        
        # 创建并分析FastSpeech2FNet模型
        print("\n创建FastSpeech2FNet模型...")
        fs2_fnet_model = FastSpeech2FNet(preprocess_config, model_config)
        analyze_model(fs2_fnet_model, "FastSpeech2FNet")
        
        # 打印模型参数量对比
        fs2_params = count_parameters(fs2_model)
        fs2_fnet_params = count_parameters(fs2_fnet_model)
        print("\n模型参数量对比:")
        print(f"FastSpeech2: {fs2_params:,}")
        print(f"FastSpeech2FNet: {fs2_fnet_params:,}")
        print(f"参数量差异: {abs(fs2_params - fs2_fnet_params):,}")
        print(f"参数量变化比例: {((fs2_fnet_params - fs2_params) / fs2_params * 100):.2f}%")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 