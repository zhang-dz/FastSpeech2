import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model.fastspeech2_fnet import FastSpeech2FNet
from model.loss import FastSpeech2Loss
from model.optimizer import ScheduledOptim
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("初始化训练...")
    preprocess_config, model_config, train_config = configs

    # 创建输出目录
    output_directory = os.path.join(train_config["path"]["ckpt_path"], "fnet")
    os.makedirs(output_directory, exist_ok=True)
    
    # 创建日志目录
    log_directory = os.path.join(train_config["path"]["log_path"], "fnet")
    os.makedirs(log_directory, exist_ok=True)
    
    # 创建结果目录
    result_directory = os.path.join(train_config["path"]["result_path"], "fnet")
    os.makedirs(result_directory, exist_ok=True)
    
    # 获取数据集
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
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

    # 准备模型
    model = FastSpeech2FNet(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        # 尝试加载原始FastSpeech2模型的权重
        try:
            model.load_state_dict(ckpt["model"])
            print(f"成功加载FastSpeech2模型权重: {ckpt_path}")
        except:
            # 如果加载失败，尝试加载部分权重
            print(f"无法完全加载FastSpeech2模型权重，尝试加载部分权重...")
            model_dict = model.state_dict()
            new_state_dict = {}
            for k, v in ckpt["model"].items():
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载 {len(new_state_dict)}/{len(model_dict)} 个参数")
    
    # 打印模型参数数量
    num_param = get_param_num(model)
    print("模型参数数量:", num_param)

    # 准备损失函数
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    
    # 准备优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        train_config["optimizer"]["learning_rate"],
        betas=train_config["optimizer"]["betas"],
        eps=train_config["optimizer"]["eps"],
        weight_decay=train_config["optimizer"]["weight_decay"],
    )
    scheduled_optim = ScheduledOptim(
        optimizer,
        train_config["optimizer"]["learning_rate"],
        train_config["optimizer"]["warm_up_step"],
        args.restore_step,
    )
    
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
                    scheduled_optim.step_and_update_lr()
                    scheduled_optim.zero_grad()

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
                        figs=fig,
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
                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(output_directory, "{}.pth.tar".format(step)),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


"""def evaluate(model, step, configs, logger, vocoder):
    preprocess_config, model_config, train_config = configs
    
    # 获取验证集
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # 评估
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    losses = [0, 0, 0, 0, 0, 0]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                batch_losses = loss_fn(batch, output)
                for i in range(len(losses)):
                    losses[i] += batch_losses[i].item() * len(batch[0])

    losses = [l / len(dataset) for l in losses]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        step, *losses
    )

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
        figs=fig,
        tag="Validation/step_{}_{}".format(step, tag),
    )
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    log(
        logger,
        step,
        audio=wav_reconstruction,
        sampling_rate=sampling_rate,
        tag="Validation/step_{}_{}_reconstructed".format(step, tag),
    )
    log(
        logger,
        step,
        audio=wav_prediction,
        sampling_rate=sampling_rate,
        tag="Validation/step_{}_{}_synthesized".format(step, tag),
    )

    return message"""


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
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs) 