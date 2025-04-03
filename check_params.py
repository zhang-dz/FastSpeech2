import torch
import yaml
import argparse
from model.fastspeech2 import FastSpeech2
from model.fastspeech2_fnet import FastSpeech2FNet
from utils.model import get_param_num

def main():
    # 加载配置文件
    preprocess_config = yaml.load(
        open("config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader
    )
    
    # 创建FastSpeech2模型
    print("创建FastSpeech2模型...")
    fs2_model = FastSpeech2(preprocess_config, model_config)
    fs2_params = get_param_num(fs2_model)
    print(f"FastSpeech2模型参数数量: {fs2_params:,}")
    
    # 创建FastSpeech2FNet模型
    print("\n创建FastSpeech2FNet模型...")
    fs2_fnet_model = FastSpeech2FNet(preprocess_config, model_config)
    fs2_fnet_params = get_param_num(fs2_fnet_model)
    print(f"FastSpeech2FNet模型参数数量: {fs2_fnet_params:,}")
    
    # 计算参数减少的百分比
    param_reduction = (fs2_params - fs2_fnet_params) / fs2_params * 100
    print(f"\n参数减少: {fs2_params - fs2_fnet_params:,} ({param_reduction:.2f}%)")
    
    # 分析每个组件的参数
    print("\n===== FastSpeech2模型组件参数分析 =====")
    analyze_model_components(fs2_model)
    
    print("\n===== FastSpeech2FNet模型组件参数分析 =====")
    analyze_model_components(fs2_fnet_model)
    
    # 分析自注意力层和傅里叶变换层的参数
    print("\n===== 自注意力层与傅里叶变换层参数比较 =====")
    analyze_attention_vs_fourier(fs2_model, fs2_fnet_model)

def analyze_model_components(model):
    """分析模型各组件的参数数量"""
    # 编码器参数
    encoder_params = sum(p.numel() for name, p in model.named_parameters() if "encoder" in name)
    print(f"编码器参数: {encoder_params:,}")
    
    # 解码器参数
    decoder_params = sum(p.numel() for name, p in model.named_parameters() if "decoder" in name)
    print(f"解码器参数: {decoder_params:,}")
    
    # 变分适配器参数
    adaptor_params = sum(p.numel() for name, p in model.named_parameters() if "variance_adaptor" in name)
    print(f"变分适配器参数: {adaptor_params:,}")
    
    # 后处理网络参数
    postnet_params = sum(p.numel() for name, p in model.named_parameters() if "postnet" in name)
    print(f"后处理网络参数: {postnet_params:,}")
    
    # 其他参数
    other_params = sum(p.numel() for name, p in model.named_parameters() 
                      if not any(x in name for x in ["encoder", "decoder", "variance_adaptor", "postnet"]))
    print(f"其他参数: {other_params:,}")

def analyze_attention_vs_fourier(fs2_model, fs2_fnet_model):
    """比较自注意力层和傅里叶变换层的参数"""
    # FastSpeech2中的自注意力层参数
    attn_params = sum(p.numel() for name, p in fs2_model.named_parameters() if "slf_attn" in name)
    print(f"FastSpeech2自注意力层参数: {attn_params:,}")
    
    # FastSpeech2FNet中的傅里叶变换层参数
    fourier_params = sum(p.numel() for name, p in fs2_fnet_model.named_parameters() if "fourier" in name)
    print(f"FastSpeech2FNet傅里叶变换层参数: {fourier_params:,}")
    
    # 参数减少
    param_reduction = attn_params - fourier_params
    print(f"自注意力层替换为傅里叶变换层减少的参数: {param_reduction:,}")

if __name__ == "__main__":
    main() 