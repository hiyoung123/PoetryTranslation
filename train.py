#!usr/bin/env python
#-*- coding:utf-8 -*-


import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model.seq2seq_beamsearch import Encoder, Decoder, Seq2Seq
from utils import load_dataset

from sklearn.model_selection import train_test_split

torch.manual_seed(123) #保证每次运行初始化的随机数相同
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, source_dict, target_dict):
    model.eval()
    pad = target_dict['PAD']
    eos_id = target_dict['EOS']
    output_list = []
    total_loss = 0
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            # src, len_src = batch.src
            # trg, len_trg = batch.trg
            # src, len_src = batch[0], batch[1]
            # trg, len_trg = batch[2], batch[3]
            src = torch.from_numpy(batch[0]).to(device).long()
            # src, trg = src.cuda(), trg.cuda()
            trg = torch.from_numpy(batch[2]).to(device).long()
            src = Variable(src.data.to(device))
            trg = Variable(trg.data.to(device))
            output = model(src, trg, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                              trg[1:].contiguous().view(-1),
                              ignore_index=pad)
            decoded_batch = model.decode(src, trg, method='beam-search')
            output_list.extend(decoded_batch.cpu().numpy())
            total_loss += loss.data.item()
    for o in output_list:
        result = []
        for i in o:
            if i == eos_id:
                break
            result.append(target_dict[i])
        print(result)
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, source_dict, target_dict):
    model.train()
    total_loss = 0
    pad = target_dict['PAD']
    for b, batch in enumerate(train_iter):
        # print(len(batch)) # 4
        # print(len(batch[0])) # 32 batch_size
        # src, len_src = batch[0], batch[1]
        # trg, len_trg = batch[2], batch[3]
        src = torch.from_numpy(batch[0]).to(device).long()
        # src, trg = src.cuda(), trg.cuda()
        trg = torch.from_numpy(batch[2]).to(device).long()
        optimizer.zero_grad()
        try:
            output = model(src, trg)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                              trg[1:].contiguous().view(-1),
                              ignore_index=pad)
            loss.backward()
        except Exception as e:
            print(e)
            print(src.size())
            print(trg.size())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()
        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    # train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    dataset, source_dict, target_dict = load_dataset(args.batch_size)
    train_iter, val_iter = train_test_split(dataset, test_size=0.2, random_state=12345)
    source_vob_size, target_vob_size = len(source_dict[0]), len(target_dict[0])
    # print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
    #       % (len(train_iter), len(train_iter.dataset),
    #          len(test_iter), len(test_iter.dataset)))
    print("[source_vob_size]:%d [target_vob_size]:%d" % (source_vob_size, target_vob_size))

    print("[!] Instantiating models...")
    encoder = Encoder(source_vob_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, target_vob_size,
                      n_layers=1, dropout=0.0)
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, train_iter,
              target_vob_size, args.grad_clip, source_dict[0], target_dict[0])
        val_loss = evaluate(seq2seq, val_iter, target_vob_size, source_dict[0], target_dict[0])
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            torch.save(seq2seq.state_dict(), 'seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, dataset, target_vob_size, source_dict[0], target_dict[0])
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
