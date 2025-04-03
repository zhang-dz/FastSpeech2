import re
import argparse
import yaml
import torch
import numpy as np
from string import punctuation

from utils.model import get_vocoder
from utils.tools import to_device, synth_samples
from model.fastspeech2_fnet import FastSpeech2FNet
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = preprocess_config["preprocessing"]["text"]["g2p"]
    if g2p == "g2p_en":
        from g2p_en import G2p
        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
    elif g2p == "pypinyin":
        from pypinyin import pinyin, Style
        phones = []
        pinyins = pinyin(text, style=Style.TONE3)
        for p in pinyins:
            phones += p[0]
        phones = "{" + " ".join(phones) + "}"
    else:
        raise Exception("g2p method not supported")

    phones = phones.replace("sp", "")
    phones = phones.replace("{", "")
    phones = phones.replace("}", "")
    phones = phones.strip()
    
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                f"fnet_{step}",
            )


def main(args):
    # 读取配置文件
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(args.model_config, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config, "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)

    # 准备模型
    model = FastSpeech2FNet(preprocess_config, model_config).to(device)
    ckpt = torch.load(args.restore_step, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_ = False
    print(f"模型已加载，步骤: {args.restore_step}")

    # 准备声码器
    vocoder = get_vocoder(model_config, device)

    # 准备文本输入
    if args.mode == "single":
        # 处理单个文本
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError("目前只支持英文文本")
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    
    # 合成语音
    synthesize(
        model, 
        args.restore_step, 
        configs, 
        vocoder, 
        batchs, 
        (args.pitch_control, args.energy_control, args.duration_control)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["single"], default="single"
    )
    parser.add_argument(
        "--text", type=str, default="This is a test sentence for FastSpeech2 with FNet."
    )
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument(
        "--preprocess_config", type=str, default="config/LJSpeech/preprocess.yaml"
    )
    parser.add_argument(
        "--model_config", type=str, default="config/LJSpeech/model.yaml"
    )
    parser.add_argument(
        "--train_config", type=str, default="config/LJSpeech/train.yaml"
    )
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    
    args = parser.parse_args()
    
    main(args) 