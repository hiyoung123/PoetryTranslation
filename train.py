#!usr/bin/env python
#-*- coding:utf-8 -*-

import os
import numpy as np
import jieba
import pandas as pd
import torch
from collections import Counter

from model.seq2seq_attention import Seq2Seq, Encoder, Decoder
from model.criterion import MaskCriterion


torch.manual_seed(123) #保证每次运行初始化的随机数相同
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data(file_name, source_file, target_file):
    data = pd.read_csv(file_name)
    source = open(source_file, 'w', encoding='utf-8')
    target = open(target_file, 'w', encoding='utf-8')
    for i, item in data.iterrows():
        content = item['content']
        translation = item['translation']
        if str(content) == 'nan' or str(translation) == 'nan' \
                or '（' in content or '（' in translation \
                or '(' in content or '(' in translation \
                or '[' in content or '[' in translation \
                or '【' in content or '【' in translation:
            continue
        content = content.strip().replace('。', '？').replace('！', '？').split('？')
        translation = translation.strip().replace('。', '？').replace('！', '？').split('？')

        if len(content) != len(translation):
            continue

        for s, t in zip(content, translation):
            source.write(s.strip()+'\n')
            target.write(t.strip()+'\n')
        print('process len:{}'.format(i+1))
    source.close()
    target.close()


def load_data(file_name):
    #逐句读取文本，并将句子进行分词，且在句子前面加上'BOS'表示句子开始，在句子末尾加上'EOS'表示句子结束
    datas = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if(i>200000):
                break
            line = line.strip()
            if len(line) == 0:
                continue
            datas.append(["BOS"] + list(jieba.cut(line, cut_all=False)) + ["EOS"])
            print('process len:{}'.format(i+1))
    return datas


source_path = "data/source.txt"
target_path = "data/target.txt"
process_data('data/PoetryTranslation.csv', source_path, target_path)
source = load_data(source_path)
target = load_data(target_path)


def create_dict(sentences, max_words):
    # 统计文本中每个词出现的频数，并用出现次数最多的max_words个词创建词典，
    # 且在词典中加入'UNK'表示词典中未出现的词，'PAD'表示后续句子中添加的padding（保证每个batch中的句子等长）
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1

    most_common_words = word_count.most_common(max_words)  # 最常见的max_words个词
    total_words = len(most_common_words) + 2  # 总词量（+2：词典中添加了“UNK”和“PAD”）
    word_dict = {w[0]: index + 2 for index, w in enumerate(most_common_words)}  # word2index
    word_dict["UNK"] = 0
    word_dict["PAD"] = 1
    return word_dict, total_words


# word2index
source_dict, source_total_words = create_dict(sentences=source, max_words=50000)
target_dict, target_total_words = create_dict(sentences=target, max_words=50000)

# index2word
inv_source_dict = {v: k for k, v in source_dict.items()}
inv_target_dict = {v: k for k, v in target_dict.items()}


def encode(en_sentences, cn_sentences, en_dict, cn_dict, sorted_by_len):
    # 句子编码：将句子中的词转换为词表中的index

    # 不在词典中的词用”UNK“表示
    out_en_sentences = [[en_dict.get(w, en_dict['UNK']) for w in sentence] for sentence in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, cn_dict['UNK']) for w in sentence] for sentence in cn_sentences]

    # 基于英文句子的长度进行排序，返回排序后句子在原始文本中的下标
    # 目的：为使每个batch中的句子等长时，需要加padding；长度相近的放入一个batch，可使得添加的padding更少
    if (sorted_by_len):
        sorted_index = sorted(range(len(out_en_sentences)), key=lambda idx: len(out_en_sentences[idx]))
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences


source_datas, target_datas = encode(source, target, source_dict, target_dict, sorted_by_len=True)


def get_batches(num_sentences, batch_size, shuffle=True):
    # 用每个句子在原始文本中的（位置）行号创建每个batch的数据索引
    batch_first_idx = np.arange(start=0, stop=num_sentences, step=batch_size)  # 每个batch中第一个句子在文本中的位置（行号）
    if (shuffle):
        np.random.shuffle(batch_first_idx)

    batches = []
    for first_idx in batch_first_idx:
        batch = np.arange(first_idx, min(first_idx + batch_size, num_sentences), 1)  # 每个batch中句子的位置（行号）
        batches.append(batch)
    return batches


def add_padding(batch_sentences):
    # 为每个batch的数据添加padding，并记录下句子原本的长度
    lengths = [len(sentence) for sentence in batch_sentences]  # 每个句子的实际长度
    max_len = np.max(lengths)  # 当前batch中最长句子的长度
    data = []
    for sentence in batch_sentences:
        sen_len = len(sentence)
        # 将每个句子末尾添0，使得每个batch中的句子等长（后续将每个batch数据转换成tensor时，每个batch中的数据维度必须一致）
        sentence = sentence + [0] * (max_len - sen_len)
        data.append(sentence)
    data = np.array(data).astype('int32')
    data_lengths = np.array(lengths).astype('int32')
    return data, data_lengths


def generate_dataset(en, cn, batch_size):
    # 生成数据集
    batches = get_batches(len(en), batch_size)
    datasets = []
    for batch in batches:
        batch_en = [en[idx] for idx in batch]
        batch_cn = [cn[idx] for idx in batch]
        batch_x, batch_x_len = add_padding(batch_en)
        batch_y, batch_y_len = add_padding(batch_cn)
        datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
    return datasets


batch_size = 8
datasets = generate_dataset(source_datas, target_datas, batch_size)


dropout = 0.2
embed_size = 50
enc_hidden_size = 100
dec_hidden_size = 200
encoder = Encoder(vocab_size=source_total_words,
                  embed_size=embed_size,
                  enc_hidden_size=enc_hidden_size,
                  dec_hidden_size=dec_hidden_size,
                  directions=2,
                  dropout=dropout)
decoder = Decoder(vocab_size=target_total_words,
                  embed_size=embed_size,
                  enc_hidden_size=enc_hidden_size,
                  dec_hidden_size=dec_hidden_size,
                  dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_func = MaskCriterion().to(device)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def test(mode, data):
    model.eval()
    total_words = 0
    total_loss = 0.
    with torch.no_grad():
        for i, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()

            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len - 1).to(device).long()
            batch_y_len[batch_y_len <= 0] = 1

            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)

            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device)[None, :] < batch_y_len[:, None]
            batch_target_masks = batch_target_masks.float()

            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)

            num_words = torch.sum(batch_y_len).item()
            total_loss += loss.item() * num_words
            total_words += num_words
        print("Test Loss:", total_loss / total_words)
    return total_loss / total_words


def train(model, data, epoches):
    print('Start training')
    test_datasets = []
    best_loss = 10000
    for epoch in range(epoches):
        model.train()
        total_words = 0
        total_loss = 0.
        for it, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            # 创建验证数据集
            if (epoch != 0 and it % 10 == 0):
                test_datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
                continue
            elif (it % 10 == 0):
                continue
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()

            # 因为训练（或验证）时，decoder根据上一步的输出（预测词）和encoder_out经attention的加权和，以及上一步输出对应的实际词预测下一个词
            # 所以输入到decoder中的目标语句为[BOS, word_1, word_2, ..., word_n]
            # 预测的实际标签为[word_1, word_2, ..., word_n, EOS]
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len - 1).to(device).long()
            batch_y_len[batch_y_len <= 0] = 1

            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)

            # 生成masks：
            batch_y_len = batch_y_len.unsqueeze(1)  # [batch_size, 1]
            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device) < batch_y_len
            batch_target_masks = batch_target_masks.float()
            batch_y_len = batch_y_len.squeeze(1)  # [batch_size]

            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)

            num_words = torch.sum(batch_y_len).item()  # 每个batch总的词量
            total_loss += loss.item() * num_words
            total_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if (it % 50 == 0):
                print("Epoch {} / {}, Iteration: {}, Train Loss: {}".format(epoch, epoches, it, loss.item()))
        print("Epoch {} / {}, Train Loss: {}".format(epoch, epoches, total_loss / total_words))
        if (epoch != 0 and epoch % 100 == 0):
            val_loss = test(model, test_datasets)
            if val_loss < best_loss:
                best_loss = val_loss


def save(model, file_path):
    # torch.save(self.model.cpu(), file_path)
    # self.model.to(self.device)
    torch.save(model.state_dict(), file_path)
    print('Model save {}'.format(file_path))


def load(model, file_path):
    if not os.path.exists(file_path):
        return
    # self.model = torch.load(file_path)
    model.load_state_dict(torch.load(file_path))

train(model, datasets, epoches=200)


def translate(sentence_id):
    # 英文翻译成中文
    source_sentence = " ".join([inv_source_dict[w] for w in source_datas[sentence_id]])  # 英文句子
    target_sentence = " ".join([inv_target_dict[w] for w in target_datas[sentence_id]])  # 对应实际的中文句子

    batch_x = torch.from_numpy(np.array(source_datas[sentence_id]).reshape(1, -1)).to(device).long()
    batch_x_len = torch.from_numpy(np.array([len(source_datas[sentence_id])])).to(device).long()

    # 第一个时间步的前项输出
    bos = torch.Tensor([[target_dict["BOS"]]]).to(device).long()

    translation = model.translate(batch_x, batch_x_len, bos, 10)
    translation = [inv_target_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]  # index2word

    trans = []
    for word in translation:
        if (word != "EOS"):
            trans.append(word)
        else:
            break
    print(source_sentence)
    print(target_sentence)
    print(" ".join(trans))

translate(0)