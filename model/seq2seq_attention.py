#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(123) #保证每次运行初始化的随机数相同
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, directions, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # the input size of gru is [sentence_len, batch_size, word_embedding_size]
        # if batch_first=True  => [batch_size, sentence_len, word_embedding_size]
        self.gru = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=(directions == 2))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, batch_x, lengths):
        # batch_x: [batch_size, max_x_setence_len]
        # lengths: [batch_size]

        # 基于每个batch中句子的实际长度倒序（后续使用pad_packed_sequence要求句子长度需要倒排序）
        sorted_lengths, sorted_index = lengths.sort(0, descending=True)
        batch_x_sorted = batch_x[sorted_index.long()]

        embed = self.embedding(batch_x_sorted)  # [batch_size, max_x_sentence_len, embed_size]
        embed = self.dropout(embed)

        # 将句子末尾添加的padding去掉，使得GRU只对实际有效语句进行编码
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(),
                                                         batch_first=True)
        packed_out, hidden = self.gru(
            packed_embed)  # packed_out为PackedSequence类型数据，hidden为tensor类型:[2, batch_size, enc_hidden_size]

        # unpacked，恢复数据为tensor
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out,
                                                  batch_first=True)  # [batch_size, max_x_sentence_len, enc_hidden_size * 2]

        # 恢复batch中sentence原始的顺序
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()

        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # [batch_size, enc_hidden_size*2]

        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  # [1, batch_size, dec_hidden_size]

        return out, hidden  # [batch_size, max_x_sentence_len, enc_hidden_size*2], [1, batch_size, dec_hidden_size]


# attention
class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = encoder_hidden_size
        self.dec_hidden_size = decoder_hidden_size

        self.linear_in = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size, bias=False)
        self.linear_out = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size)

    def forward(self, output, context, masks):
        # output [batch_size, max_y_sentence_len, dec_hidden_size]
        # context [batch_size, max_x_sentence_len, enc_hidden_size*2]
        # masks [batch_size, max_y_sentence_len, max_x_sentence_len]

        batch_size = output.size(0)
        y_len = output.size(1)
        x_len = context.size(1)

        x = context.view(batch_size * x_len, -1)  # [batch_size * max_x_sentence_len, enc_hidden_size*2]
        x = self.linear_in(x)  # [batch_size * max_x_len, dec_hidden_size]

        context_in = x.view(batch_size, x_len, -1)  # [batch_size, max_x_sentence_len, dec_hidden_size]
        atten = torch.bmm(output, context_in.transpose(1, 2))  # [batch_size, max_y_sentence_len, max_x_sentence_len]

        atten.data.masked_fill_(masks.bool(), -1e-6)

        atten = F.softmax(atten, dim=2)  # [batch_size, max_y_sentence_len, max_x_sentence_len]

        context = torch.bmm(atten, context)  # [batch_size, max_y_sentence_len, enc_hidden_size*2]
        output = torch.cat((context, output),
                           dim=2)  # [batch_size, max_y_sentence_len, enc_hidden_size*2+dec_hidden_size]

        output = output.view(batch_size * y_len,
                             -1)  # [batch_size * max_y_sentence_len, enc_hidden_size*2+dec_hidden_size]
        output = torch.tanh(self.linear_out(output))

        output = output.view(batch_size, y_len, -1)  # [batch_size, max_y_sentence_len, dec_hidden_size]

        return output, atten


# seq2seq的解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.gru = nn.GRU(embed_size, dec_hidden_size, batch_first=True)
        # 将每个输出都映射会词表维度，最大值所在的位置对应的词就是预测的目标词
        self.liner = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_atten_masks(self, x_len, y_len):
        # 创建attention的masks
        # 超出句子有效长度部分的attention用一个很小的数填充，使其在softmax后的权重很小
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_masks = torch.arange(max_x_len, device=device)[None, :] < x_len[:, None]  # [batch_size, max_x_sentence_len]
        y_masks = torch.arange(max_y_len, device=device)[None, :] < y_len[:, None]  # [batch_size, max_y_sentence_len]

        # x_masks[:, :, None] [batch_size, max_x_sentence_len, 1]
        # y_masks[:, None, :][batch_size, 1, max_y_sentence_len]
        # masked_fill_填充的是True所在的维度，所以取反(~)
        masks = (
            ~(y_masks[:, :, None] * x_masks[:, None, :])).byte()  # [batch_size, max_y_sentence_len, max_x_sentence_len]

        return masks  # [batch_size, max_y_sentence_len, max_x_sentence_len]

    def forward(self, encoder_out, x_lengths, batch_y, y_lengths, encoder_hidden):
        # batch_y: [batch_size, max_x_setence_len]
        # lengths: [batch_size]
        # encoder_hidden: [1, batch_size, dec_hidden_size*2]

        # 基于每个batch中句子的实际长度倒序
        sorted_lengths, sorted_index = y_lengths.sort(0, descending=True)
        batch_y_sorted = batch_y[sorted_index.long()]
        hidden = encoder_hidden[:, sorted_index.long()]

        embed = self.embedding(batch_y_sorted)  # [batch_size, max_x_setence_len, embed_size]
        embed = self.dropout(embed)

        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(),
                                                         batch_first=True)
        # 解码器的输入为编码器的输出，上一个词，然后预测下一个词
        packed_out, hidden = self.gru(packed_embed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()  # [batch_size, max_y_sentence_len, dec_hidden_size]
        hidden = hidden[:, original_index.long()].contiguous()  # [1, batch_size, dec_hidden_size]

        atten_masks = self.create_atten_masks(x_lengths,
                                              y_lengths)  # [batch_size, max_y_sentcnec_len, max_x_sentcnec_len]

        out, atten = self.attention(out, encoder_out,
                                    atten_masks)  # out [batch_size, max_y_sentence_len, dec_hidden_size]

        out = self.liner(out)  # [batch_size, cn_sentence_len, vocab_size]

        # log_softmax求出每个输出的概率分布，最大概率出现的位置就是预测的词在词表中的位置
        out = F.log_softmax(out, dim=-1)  # [batch_size, cn_sentence_len, vocab_size]
        return out, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, encoder_hid = self.encoder(x, x_lengths)  # 源语言编码
        output, hidden = self.decoder(encoder_out, x_lengths, y, y_lengths, encoder_hid)  # 解码出目标
        return output

    def translate(self, x, x_lengths, y, max_length=50):
        # 翻译en2cn
        # max_length表示翻译的目标句子可能的最大长度
        encoder_out, encoder_hidden = self.encoder(x, x_lengths)  # 将输入的英文进行编码
        predicts = []
        batch_size = x.size(0)
        # 目标语言（中文）的输入只有”BOS“表示句子开始，因此y的长度为1
        # 每次都用上一个词(y)与编码器的输出预测下一个词，因此y的长度一直为1
        y_length = torch.ones(batch_size).long().to(y.device)
        for i in range(max_length):
            # 每次用上一次的输出y和编码器的输出encoder_hidden预测下一个词
            output, hidden = self.decoder(encoder_out, x_lengths, y, y_length, encoder_hidden)
            # output: [batch_size, 1, vocab_size]

            # output.max(2)[1]表示找出output第二个维度的最大值所在的位置（即预测词在词典中的index）
            y = output.max(2)[1].view(batch_size, 1)  # [batch_size, 1]
            predicts.append(y)

        predicts = torch.cat(predicts, 1)  # [batch_size, max_length]

        return predicts