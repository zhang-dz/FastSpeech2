import torch
import yaml
import argparse
from pathlib import Path
import sys

from model.fastspeech2_fnet import FastSpeech2FNet


def main(args):
    print("开始执行测试脚本...")
    
    try:
        # 加载配置文件
        print(f"正在加载配置文件: {args.preprocess_config}, {args.model_config}")
        preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        print("配置文件加载成功")
        
        # 创建FastSpeech2FNet模型
        print("正在创建FastSpeech2FNet模型...")
        model = FastSpeech2FNet(preprocess_config, model_config)
        print("模型创建成功")
        
        # 打印模型结构
        print("模型结构:")
        print(model)
        
        # 计算模型参数数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {num_params:,}")
        
        # 如果有预训练的FastSpeech2模型，可以加载部分权重
        if args.restore_path:
            print(f"从 {args.restore_path} 加载预训练权重...")
            ckpt = torch.load(args.restore_path, map_location="cpu")
            
            # 创建一个新的状态字典，只包含匹配的键
            new_state_dict = {}
            model_dict = model.state_dict()
            
            # 尝试加载匹配的参数
            for k, v in ckpt["model"].items():
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    print(f"加载参数: {k}")
            
            # 加载匹配的参数
            model.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载 {len(new_state_dict)}/{len(model_dict)} 个参数")
        
        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"模型已创建并加载到 {device}")
        print("现在可以使用此模型进行训练或推理")
    
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess_config", 
        type=str, 
        default="config/LJSpeech/preprocess.yaml"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="config/LJSpeech/model.yaml"
    )
    parser.add_argument(
        "--restore_path", 
        type=str, 
        default=None,
        help="预训练的FastSpeech2模型路径（可选）"
    )
    args = parser.parse_args()
    
    print(f"命令行参数: {args}")
    main(args) 