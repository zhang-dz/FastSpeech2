import os
import torch
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import yaml

from model import FastSpeech2, FastSpeech2FNet
from transformer.Models import Encoder, Decoder
from transformer.FourierModels import FourierEncoder, FourierDecoder
from transformer.Layers import FFTBlock
from transformer.FourierLayers import FourierBlock, FourierTransform
from transformer.SubLayers import MultiHeadAttention

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model_structure_image(model_name, blocks, param_count, is_fourier=False):
    """创建模型结构图像"""
    # 设置图像大小和边距
    width, height = 800, 1200
    margin = 50
    block_height = 80
    block_spacing = 40
    
    # 创建空白图像和绘图对象
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # 尝试加载中文字体
        font_title = ImageFont.truetype("simhei.ttf", 32)
        font_block = ImageFont.truetype("simhei.ttf", 20)
        font_param = ImageFont.truetype("simhei.ttf", 18)
    except IOError:
        # 如果找不到中文字体，使用默认字体
        font_title = ImageFont.load_default()
        font_block = ImageFont.load_default()
        font_param = ImageFont.load_default()
    
    # 绘制标题
    title = f"{model_name} 模型结构"
    title_width = draw.textlength(title, font=font_title)
    draw.text(((width - title_width) // 2, margin), title, fill="black", font=font_title)
    
    # 绘制参数数量
    param_text = f"参数数量: {param_count:,}"
    param_width = draw.textlength(param_text, font=font_param)
    draw.text(((width - param_width) // 2, margin + 50), param_text, fill="black", font=font_param)
    
    # 创建模型框架
    # 输入
    draw_rounded_rectangle(draw, (margin, margin + 100, width - margin, margin + 150), fill="lightblue", outline="black", width=2, radius=10)
    draw.text((width // 2 - 50, margin + 115), "输入", fill="black", font=font_block)
    
    # 词嵌入和位置编码
    draw_rounded_rectangle(draw, (margin, margin + 180, width - margin, margin + 230), fill="lightgreen", outline="black", width=2, radius=10)
    draw.text((width // 2 - 120, margin + 195), "词嵌入 + 位置编码", fill="black", font=font_block)
    
    # 编码器层
    start_y = margin + 260
    for i, (block_name, layer_params) in enumerate(blocks['encoder']):
        block_y = start_y + i * (block_height + block_spacing)
        
        # 块颜色选择
        if "注意力" in block_name:
            color = "lightcoral"  # 注意力机制为红色
        elif "傅里叶" in block_name:
            color = "lightskyblue"  # 傅里叶变换为蓝色
        else:
            color = "lightyellow"  # 前馈网络为黄色
            
        draw_rounded_rectangle(draw, (margin, block_y, width - margin, block_y + block_height), 
                       fill=color, outline="black", width=2, radius=10)
        
        # 绘制块名称和参数
        draw.text((margin + 20, block_y + 15), block_name, fill="black", font=font_block)
        draw.text((margin + 20, block_y + 45), f"参数: {layer_params:,}", fill="black", font=font_param)
    
    # 变分适配器层
    va_y = start_y + len(blocks['encoder']) * (block_height + block_spacing) + 20
    draw_rounded_rectangle(draw, (margin, va_y, width - margin, va_y + block_height), 
                   fill="lightpink", outline="black", width=2, radius=10)
    draw.text((width // 2 - 60, va_y + 25), "变分适配器", fill="black", font=font_block)
    
    # 解码器层
    start_decoder_y = va_y + block_height + 20
    for i, (block_name, layer_params) in enumerate(blocks['decoder']):
        block_y = start_decoder_y + i * (block_height + block_spacing)
        
        # 块颜色选择
        if "注意力" in block_name:
            color = "lightcoral"  # 注意力机制为红色
        elif "傅里叶" in block_name:
            color = "lightskyblue"  # 傅里叶变换为蓝色
        else:
            color = "lightyellow"  # 前馈网络为黄色
            
        draw_rounded_rectangle(draw, (margin, block_y, width - margin, block_y + block_height), 
                       fill=color, outline="black", width=2, radius=10)
        
        # 绘制块名称和参数
        draw.text((margin + 20, block_y + 15), block_name, fill="black", font=font_block)
        draw.text((margin + 20, block_y + 45), f"参数: {layer_params:,}", fill="black", font=font_param)
    
    # 线性输出层和PostNet
    linear_y = start_decoder_y + len(blocks['decoder']) * (block_height + block_spacing) + 20
    draw_rounded_rectangle(draw, (margin, linear_y, width - margin, linear_y + block_height), 
                   fill="lightgreen", outline="black", width=2, radius=10)
    draw.text((width // 2 - 120, linear_y + 25), "线性层 + PostNet", fill="black", font=font_block)
    
    # 输出
    output_y = linear_y + block_height + 40
    draw_rounded_rectangle(draw, (margin, output_y, width - margin, output_y + 50), 
                   fill="lightblue", outline="black", width=2, radius=10)
    draw.text((width // 2 - 50, output_y + 15), "输出", fill="black", font=font_block)
    
    # 添加连接线
    # 输入到词嵌入
    draw_arrow(draw, width // 2, margin + 150, width // 2, margin + 180)
    
    # 词嵌入到第一个编码器层
    draw_arrow(draw, width // 2, margin + 230, width // 2, start_y)
    
    # 编码器层连接
    for i in range(len(blocks['encoder']) - 1):
        y1 = start_y + i * (block_height + block_spacing) + block_height
        y2 = y1 + block_spacing
        draw_arrow(draw, width // 2, y1, width // 2, y2)
    
    # 最后一个编码器层到变分适配器
    draw_arrow(draw, width // 2, start_y + (len(blocks['encoder']) - 1) * (block_height + block_spacing) + block_height, 
             width // 2, va_y)
    
    # 变分适配器到第一个解码器层
    draw_arrow(draw, width // 2, va_y + block_height, width // 2, start_decoder_y)
    
    # 解码器层连接
    for i in range(len(blocks['decoder']) - 1):
        y1 = start_decoder_y + i * (block_height + block_spacing) + block_height
        y2 = y1 + block_spacing
        draw_arrow(draw, width // 2, y1, width // 2, y2)
    
    # 最后一个解码器层到线性层
    draw_arrow(draw, width // 2, start_decoder_y + (len(blocks['decoder']) - 1) * (block_height + block_spacing) + block_height, 
             width // 2, linear_y)
    
    # 线性层到输出
    draw_arrow(draw, width // 2, linear_y + block_height, width // 2, output_y)
    
    # 添加图例
    legend_y = output_y + 70
    # 注意力机制/傅里叶变换
    if is_fourier:
        draw_rounded_rectangle(draw, (margin, legend_y, margin + 30, legend_y + 30), 
                      fill="lightskyblue", outline="black", width=2, radius=5)
        draw.text((margin + 40, legend_y + 5), "傅里叶变换层", fill="black", font=font_param)
    else:
        draw_rounded_rectangle(draw, (margin, legend_y, margin + 30, legend_y + 30), 
                      fill="lightcoral", outline="black", width=2, radius=5)
        draw.text((margin + 40, legend_y + 5), "多头自注意力层", fill="black", font=font_param)
    
    # 前馈网络
    draw_rounded_rectangle(draw, (width // 2, legend_y, width // 2 + 30, legend_y + 30), 
                  fill="lightyellow", outline="black", width=2, radius=5)
    draw.text((width // 2 + 40, legend_y + 5), "前馈网络层", fill="black", font=font_param)
    
    return img

def draw_rounded_rectangle(draw, xy, radius=10, fill=None, outline=None, width=1):
    """绘制圆角矩形"""
    x1, y1, x2, y2 = xy
    # 绘制四个角
    draw.pieslice([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, fill=fill, outline=outline, width=width)
    draw.pieslice([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, fill=fill, outline=outline, width=width)
    draw.pieslice([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, fill=fill, outline=outline, width=width)
    draw.pieslice([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, fill=fill, outline=outline, width=width)
    
    # 绘制矩形
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill, outline=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill, outline=fill)
    
    # 绘制边框
    if outline:
        draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)  # 上边
        draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)  # 下边
        draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)  # 左边
        draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)  # 右边

def draw_arrow(draw, x1, y1, x2, y2, fill="black", width=2, arrow_size=10):
    """绘制箭头"""
    # 绘制线
    draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
    
    # 计算箭头方向
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length == 0:
        return
    
    # 单位向量
    udx = dx / length
    udy = dy / length
    
    # 计算箭头的两个点
    arrow_x1 = x2 - arrow_size * udx + arrow_size * udy * 0.5
    arrow_y1 = y2 - arrow_size * udy - arrow_size * udx * 0.5
    arrow_x2 = x2 - arrow_size * udx - arrow_size * udy * 0.5
    arrow_y2 = y2 - arrow_size * udy + arrow_size * udx * 0.5
    
    # 绘制箭头
    draw.polygon([(x2, y2), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)], fill=fill)

def get_model_info(model_name, model):
    """获取模型结构信息"""
    blocks = {'encoder': [], 'decoder': []}
    
    # 检查模型类型
    is_fourier = isinstance(model, FastSpeech2FNet)
    
    # 获取编码器层信息
    encoder_layers = model.encoder.layer_stack
    for i, layer in enumerate(encoder_layers):
        if is_fourier:
            # FourierBlock
            block_name = f"编码器层 {i+1}: 傅里叶变换"
            fourier_params = sum(p.numel() for p in layer.fourier.parameters() if p.requires_grad)
            ffn_params = sum(p.numel() for p in layer.pos_ffn.parameters() if p.requires_grad)
            blocks['encoder'].append((block_name, fourier_params + ffn_params))
        else:
            # FFTBlock with MultiHeadAttention
            block_name = f"编码器层 {i+1}: 多头自注意力"
            attn_params = sum(p.numel() for p in layer.slf_attn.parameters() if p.requires_grad)
            ffn_params = sum(p.numel() for p in layer.pos_ffn.parameters() if p.requires_grad)
            blocks['encoder'].append((block_name, attn_params + ffn_params))
    
    # 获取解码器层信息
    decoder_layers = model.decoder.layer_stack
    for i, layer in enumerate(decoder_layers):
        if is_fourier:
            # FourierBlock
            block_name = f"解码器层 {i+1}: 傅里叶变换"
            fourier_params = sum(p.numel() for p in layer.fourier.parameters() if p.requires_grad)
            ffn_params = sum(p.numel() for p in layer.pos_ffn.parameters() if p.requires_grad)
            blocks['decoder'].append((block_name, fourier_params + ffn_params))
        else:
            # FFTBlock with MultiHeadAttention
            block_name = f"解码器层 {i+1}: 多头自注意力"
            attn_params = sum(p.numel() for p in layer.slf_attn.parameters() if p.requires_grad)
            ffn_params = sum(p.numel() for p in layer.pos_ffn.parameters() if p.requires_grad)
            blocks['decoder'].append((block_name, attn_params + ffn_params))
    
    # 计算总参数量
    total_params = count_parameters(model)
    
    return blocks, total_params, is_fourier

def create_comparison_image(original_model, fnet_model):
    """创建模型比较图"""
    # 获取模型信息
    original_blocks, original_params, _ = get_model_info("FastSpeech2", original_model)
    fnet_blocks, fnet_params, _ = get_model_info("FastSpeech2FNet", fnet_model)
    
    # 创建模型结构图
    original_img = create_model_structure_image("FastSpeech2 (原始模型)", original_blocks, original_params, is_fourier=False)
    fnet_img = create_model_structure_image("FastSpeech2FNet (新模型)", fnet_blocks, fnet_params, is_fourier=True)
    
    # 合并图像
    total_width = original_img.width + fnet_img.width
    max_height = max(original_img.height, fnet_img.height)
    
    comparison_img = Image.new('RGB', (total_width, max_height), color='white')
    comparison_img.paste(original_img, (0, 0))
    comparison_img.paste(fnet_img, (original_img.width, 0))
    
    # 添加比较信息
    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("simhei.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    # 在图片底部添加参数对比信息
    param_diff = original_params - fnet_params
    param_percent = (param_diff / original_params) * 100
    
    info_text = f"参数量对比: 原模型 {original_params:,} vs 新模型 {fnet_params:,}"
    info_text2 = f"新模型减少了 {param_diff:,} 个参数 ({param_percent:.2f}%)"
    
    draw.text((total_width // 2 - 200, max_height - 80), info_text, fill="black", font=font)
    draw.text((total_width // 2 - 200, max_height - 40), info_text2, fill="black", font=font)
    
    return comparison_img

def generate_model_structures(preprocess_config, model_config):
    """生成模型结构图"""
    # 创建输出目录
    output_dir = "output/model_structure"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型
    original_model = FastSpeech2(preprocess_config, model_config)
    fnet_model = FastSpeech2FNet(preprocess_config, model_config)
    
    # 计算参数量
    original_params = count_parameters(original_model)
    fnet_params = count_parameters(fnet_model)
    
    print(f"原始模型 (FastSpeech2) 参数量: {original_params:,}")
    print(f"新模型 (FastSpeech2FNet) 参数量: {fnet_params:,}")
    print(f"差异: {original_params - fnet_params:,} 参数 ({(original_params - fnet_params) / original_params * 100:.2f}%)")
    
    # 生成原模型结构图
    original_blocks, _, _ = get_model_info("FastSpeech2", original_model)
    original_img = create_model_structure_image("FastSpeech2 (原始模型)", original_blocks, original_params, is_fourier=False)
    original_img.save(os.path.join(output_dir, "fastspeech2_structure.png"))
    
    # 生成新模型结构图
    fnet_blocks, _, _ = get_model_info("FastSpeech2FNet", fnet_model)
    fnet_img = create_model_structure_image("FastSpeech2FNet (新模型)", fnet_blocks, fnet_params, is_fourier=True)
    fnet_img.save(os.path.join(output_dir, "fastspeech2_fnet_structure.png"))
    
    # 生成模型比较图
    comparison_img = create_comparison_image(original_model, fnet_model)
    comparison_img.save(os.path.join(output_dir, "model_comparison.png"))
    
    print(f"模型结构图已保存到 {output_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    args = parser.parse_args()

    # 读取配置文件
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    
    # 生成模型结构图
    generate_model_structures(preprocess_config, model_config) 