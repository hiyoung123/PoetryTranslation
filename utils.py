#!usr/bin/env python
#-*- coding:utf-8 -*-

import jieba
import numpy as np
from collections import Counter

def load_data(file_name):
    # 逐句读取文本，并将句子进行分词，且在句子前面加上'BOS'表示句子开始，在句子末尾加上'EOS'表示句子结束
    datas = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if (i > 200000):
                break
            line = line.strip()
            if len(line) == 0:
                continue
            datas.append(["BOS"] + list(jieba.cut(line, cut_all=False)) + ["EOS"])
            print('process len:{}'.format(i + 1))
    return datas


def create_dict(sentences, max_words):
    # 统计文本中每个词出现的频数，并用出现次数最多的max_words个词创建词典，
    # 且在词典中加入'UNK'表示词典中未出现的词，'PAD'表示后续句子中添加的padding（保证每个batch中的句子等长）
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1

    most_common_words = word_count.most_common(max_words)  # 最常见的max_words个词
    total_words = len(most_common_words) + 2  # 总词量（+2：词典中添加了“UNK”和“PAD”）
    #word_dict = {w[0]: index + 4 for index, w in enumerate(most_common_words) if w[0] not in ['BOS', 'EOS']}  # word2index
    word_dict = {}
    i = 0
    for w in most_common_words:
        if w[0] in ['BOS', 'EOS']:
            continue
        word_dict[w[0]] = i + 4
        i = i+1
    word_dict["UNK"] = 0
    word_dict["PAD"] = 1
    word_dict["BOS"] = 2
    word_dict["EOS"] = 3
    return word_dict, total_words



def encode(source_sentences, target_sentences, source_dict, target_dict, sorted_by_len):
    # 句子编码：将句子中的词转换为词表中的index

    # 不在词典中的词用”UNK“表示
    out_source_sentences = [[source_dict.get(w, source_dict['UNK']) for w in sentence] for sentence in source_sentences]
    out_target_sentences = [[target_dict.get(w, target_dict['UNK']) for w in sentence] for sentence in target_sentences]

    # 基于英文句子的长度进行排序，返回排序后句子在原始文本中的下标
    # 目的：为使每个batch中的句子等长时，需要加padding；长度相近的放入一个batch，可使得添加的padding更少
    if (sorted_by_len):
        sorted_index = sorted(range(len(out_source_sentences)), key=lambda idx: len(out_source_sentences[idx]))
        out_source_sentences = [out_source_sentences[i] for i in sorted_index]
        out_target_sentences = [out_target_sentences[i] for i in sorted_index]

    return out_source_sentences, out_target_sentences


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


def add_padding(batch_sentences, max_len):
    # 为每个batch的数据添加padding，并记录下句子原本的长度
    # lengths = [len(sentence) for sentence in batch_sentences]  # 每个句子的实际长度
    # max_len = np.max(lengths)  # 当前batch中最长句子的长度
    data = []
    for sentence in batch_sentences:
        sen_len = len(sentence)
        # 将每个句子末尾添0，使得每个batch中的句子等长（后续将每个batch数据转换成tensor时，每个batch中的数据维度必须一致）
        sentence = sentence + [1] * (max_len - sen_len)
        data.append(sentence)
    data = np.array(data).astype('int32')
    # data_lengths = np.array(lengths).astype('int32')
    data_lengths = np.array(max_len).astype('int32')
    return data, data_lengths


def generate_dataset(en, cn, batch_size):
    # 生成数据集
    batches = get_batches(len(en), batch_size)
    datasets = []
    for batch in batches:
        batch_en = [en[idx] for idx in batch]
        batch_cn = [cn[idx] for idx in batch]
        max_len = max(map(len, batch_en+batch_cn))
        batch_x, batch_x_len = add_padding(batch_en, max_len)
        batch_y, batch_y_len = add_padding(batch_cn, max_len)
        datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
    return datasets


def load_dataset(batch_size):

    source_path = "data/source.txt"
    target_path = "data/target.txt"
    source = load_data(source_path)
    target = load_data(target_path)

    # word2index
    source_dict, source_total_words = create_dict(sentences=source, max_words=50000)
    target_dict, target_total_words = create_dict(sentences=target, max_words=50000)

    # index2word
    inv_source_dict = {v: k for k, v in source_dict.items()}
    inv_target_dict = {v: k for k, v in target_dict.items()}

    source_datas, target_datas = encode(source, target, source_dict, target_dict, sorted_by_len=True)

    datasets = generate_dataset(source_datas, target_datas, batch_size)
    return datasets, (source_dict, inv_source_dict), (target_dict, inv_target_dict)
