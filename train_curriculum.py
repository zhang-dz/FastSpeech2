import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.mmodel_fnet import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model.loss import FastSpeech2Loss
from dataloadCustom.curriculum_loader import CurriculumDataset
from evaluate_fnet import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("初始化课程学习训练...")
    preprocess_config, model_config, train_config = configs

    # 创建输出目录
    output_directory = os.path.join(train_config["path"]["ckpt_path"], "curriculum")
    os.makedirs(output_directory, exist_ok=True)
    
    # 创建日志目录
    log_directory = os.path.join(train_config["path"]["log_path"], "curriculum")
    os.makedirs(log_directory, exist_ok=True)
    
    # 创建结果目录
    result_directory = os.path.join(train_config["path"]["result_path"], "curriculum")
    os.makedirs(result_directory, exist_ok=True)
    
    # 获取数据集
    dataset = CurriculumDataset(
        "train.txt", 
        preprocess_config, 
        train_config, 
        sort=True, 
        drop_last=True,
        num_levels=train_config.get("curriculum", {}).get("num_levels", 5),
        current_level=1
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # 准备模型和优化器
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)

    # 打印模型参数数量
    num_param = get_param_num(model)
    print("模型参数数量:", num_param)

    # 准备损失函数
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    
    # 准备日志
    logger = SummaryWriter(log_directory)
    
    # 准备声码器
    vocoder = get_vocoder(model_config, device)

    # 训练
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    # 课程学习相关参数
    curriculum_config = train_config.get("curriculum", {})
    val_loss_threshold = curriculum_config.get("val_loss_threshold", 0.1)  # 相对阈值，表示损失需要改善10%才能进入下一等级
    min_steps_per_level = curriculum_config.get("min_steps_per_level", 10000)  # 每个等级最少训练步数
    max_steps_per_level = curriculum_config.get("max_steps_per_level", 100000)  # 每个等级最多训练步数
    steps_in_current_level = 0
    best_val_loss = float('inf')
    patience = curriculum_config.get("patience", 5)  # 验证损失不下降的容忍步数
    no_improve_steps = 0
    level_start_loss = None  # 记录每个难度等级开始时的损失值

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = loss_fn(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(log_directory, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        logger,
                        step,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        logger,
                        step,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        logger,
                        step,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, logger, vocoder)
                    with open(os.path.join(log_directory, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    
                    # 检查是否需要进入下一个难度等级
                    val_loss = float(message.split("Total Loss: ")[1].split(",")[0])
                    
                    # 如果是当前难度等级的第一步，记录初始损失
                    if level_start_loss is None:
                        level_start_loss = val_loss
                        best_val_loss = val_loss
                    
                    # 更新最佳验证损失
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    
                    # 计算相对于初始损失的改善比例
                    improvement_ratio = (level_start_loss - val_loss) / level_start_loss
                    
                    # 检查是否满足进入下一等级的条件
                    steps_in_current_level += val_step
                    if (steps_in_current_level >= min_steps_per_level and  # 满足最小训练步数
                        improvement_ratio >= val_loss_threshold and  # 损失改善达到阈值
                        (no_improve_steps >= patience or  # 验证损失不再下降
                         steps_in_current_level >= max_steps_per_level)):  # 达到最大训练步数
                        
                        if dataset.advance_level(val_loss):
                            print(f"\n进入难度等级 {dataset.current_level}")
                            print("当前等级统计信息:", dataset.get_level_stats())
                            print(f"损失改善比例: {improvement_ratio:.4f}")
                            steps_in_current_level = 0
                            best_val_loss = float('inf')
                            no_improve_steps = 0
                            level_start_loss = None  # 重置初始损失
                    
                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "current_level": dataset.current_level,
                            "steps_in_current_level": steps_in_current_level,
                            "best_val_loss": best_val_loss,
                            "no_improve_steps": no_improve_steps,
                            "level_start_loss": level_start_loss,
                        },
                        os.path.join(output_directory, "{}.pth.tar".format(step)),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


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
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # 读取配置文件
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs) 