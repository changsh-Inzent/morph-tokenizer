import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.tensorboard import SummaryWriter

import json
import argparse
import datetime
import os
from collections import Counter
import math

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.set_num_threads(8)
torch.set_num_interop_threads(8)
print(torch.get_num_threads())

class RestoreDataset(IterableDataset):
    UNK_INDEX = 0
    PAD_INDEX = 1
    BOS_INDEX = 2
    EOS_INDEX = 3

    def __init__(self, corpus_filename, max_characters, source_vocab=None, target_vocab=None):
        super(RestoreDataset, self).__init__()

        self.corpus_filename = corpus_filename
        self.max_characters = max_characters

        # 이미 만들어진 vocab을 사용하는 경우는 corpus에서 vocab을 만들지 않습니다.
        # 예를 들어 이미 저장된 checkpoint부터 훈련을 다시 사용할 경우가 그렇습니다.
        if source_vocab:
            self.source_vocab = source_vocab
            self.target_vocab = target_vocab
        else:
            print('Building a vocab')
            self.source_vocab, self.target_vocab = self.build_vocab()
            print(f'Found {len(self.source_vocab)} source tokens')        
            print(f'Found {len(self.target_vocab)} target tokens')        

    def build_vocab(self):
        source_counter = Counter()
        target_counter = Counter()

        for line in open(self.corpus_filename):            
            data = json.loads(line)
            # Tokenizer를 사용하지 않고 글자가 각 Vocab이 됩니다.
            source_counter.update(data[0])
            # Output의 Vocab도 글자 단위지만 자음, 모음과 같이 입력에는 없는 Vocab이 있을 수도 있습니다.
            target_counter.update(data[1])

        # 특수 Vocab들을 추가합니다.
        return vocab(source_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>']), vocab(target_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    # Corpus에서 한 문장씩 읽은 후에 tensor 형태로 만들어서 돌려줍니다.
    def __iter__(self):
        for line in open(self.corpus_filename):
            data = json.loads(line)

            # 너무 긴 문장은 생략
            if len(data[0]) > self.max_characters:
                continue

            x = torch.tensor([self.source_vocab[c] for c in data[0].strip()], dtype=torch.long)
            y = torch.tensor([self.target_vocab[c] for c in data[1].strip()], dtype=torch.long)

            yield x, y
            
class MorphemeRestoreModel(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embedding_dim, source_vocab_size, target_vocab_size, num_heads, feedforward_dim=512, dropout=0.1):
        super(MorphemeRestoreModel, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=feedforward_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=feedforward_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(embedding_dim, target_vocab_size)
        self.source_embedding = TokenEmbedding(source_vocab_size, embedding_dim)
        self.target_embedding = TokenEmbedding(target_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout)

    def forward(self, source, target, source_mask, target_mask, source_padding_mask, target_padding_mask, memory_key_padding_mask):
        source_embedding = self.positional_encoding(self.source_embedding(source))
        target_embedding = self.positional_encoding(self.target_embedding(target))
        memory = self.transformer_encoder(source_embedding, source_mask, source_padding_mask)
        outs = self.transformer_decoder(target_embedding, memory, target_mask, None, target_padding_mask, memory_key_padding_mask)

        return self.generator(outs)

    def encode(self, source, source_mask):
        return self.transformer_encoder(self.positional_encoding(self.source_embedding(source)), source_mask)

    def decode(self, target, memory, target_mask):
        return self.transformer_decoder(self.positional_encoding(self.target_embedding(target)), memory, target_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == RestoreDataset.PAD_INDEX).transpose(0, 1)
    tgt_padding_mask = (tgt == RestoreDataset.PAD_INDEX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
def make_batch(data_batch):
    inputs = []
    labels = []

    for input_item, label_item in data_batch:
        inputs.append(torch.cat([torch.tensor([RestoreDataset.BOS_INDEX]), input_item, torch.tensor([RestoreDataset.EOS_INDEX])], dim=0))
        labels.append(torch.cat([torch.tensor([RestoreDataset.BOS_INDEX]), label_item, torch.tensor([RestoreDataset.EOS_INDEX])], dim=0))

    input_batch = pad_sequence(inputs, padding_value=RestoreDataset.PAD_INDEX)
    label_batch = pad_sequence(labels, padding_value=RestoreDataset.PAD_INDEX)

    return input_batch, label_batch
    
def train(epoch, model, train_dataloader, source_vocab, target_vocab, loss_fn, optimizer):
    model.train()

    total_loss = 0.0

    begin_time_batch_print = datetime.datetime.now()

    for num_batches, (x, y) in enumerate(train_dataloader, 1):
        x = x.to(DEVICE)    
        y = y.to(DEVICE)

        # Teacher Enforcing을 위해서 Decoder 입력으로 마지막 토근(<eos>)을 제외한 값을 사용합니다.
        y_input = y[:-1, :]

        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(x, y_input)
        source_mask = source_mask.to(DEVICE)
        target_mask = target_mask.to(DEVICE)
        source_padding_mask = source_padding_mask.to(DEVICE)
        target_padding_mask = target_padding_mask.to(DEVICE)

        logits = model(x, y_input, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)

        optimizer.zero_grad()

        # Decoder의 출력은 두번째 토큰부터 <eos>까지여야 합니다.
        y_out = y[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y_out.reshape(-1))
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if num_batches % args.batch_print == 0:
            end_time_batch_print = datetime.datetime.now()   
            print(f'Epoch {epoch:>2d},\tBatch {num_batches:>6d},\tloss: {total_loss / num_batches:.10f}\tTook {str(end_time_batch_print - begin_time_batch_print)}')

            begin_time_batch_print = datetime.datetime.now()

    # 나중에 Checkpoint 부터 훈련이 재개할 때를 대비해서 필요한 정보를 저장합니다.
    print('Saving checkpoint')
    torch.save({
        'epoch': epoch,
        'source_vocab_size': len(source_vocab),
        'target_vocab_size': len(target_vocab),
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'feedforward_dim': args.feedforward_dim,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        'source_vocab': source_vocab,
        'target_vocab': target_vocab
    }, os.path.join(args.output, f'model_{epoch:02d}.pth'))
    print('Done')

    return total_loss / num_batches
    
if __name__ == '__main__':
    print(f'Using {DEVICE}')

    # 각종 hyperprameter를 명령행으로 받을 수 있도록 준비합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True)
    parser.add_argument('-b', '--batch', type=int, required=True)
    parser.add_argument('-e', '--epoch', type=int, required=True)
    parser.add_argument('-ed', '--embedding-dim', type=int)
    parser.add_argument('-nel', '--num-encoder-layers', type=int)
    parser.add_argument('-ndl', '--num-decoder-layers', type=int)
    parser.add_argument('-nh', '--num-heads', type=int)
    parser.add_argument('-fd', '--feedforward-dim', type=int)
    parser.add_argument('-bp', '--batch-print', type=int, required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-r', '--resume', )
    parser.add_argument('-m', '--max-characters', type=int)

    args = parser.parse_args()

    # tensorboard를 통해서 loss를 추적합니다.
    writer = SummaryWriter()

    os.makedirs(args.output, exist_ok=True)

    # loss를 계산할 때 Padding 값은 무시합니다.
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=RestoreDataset.PAD_INDEX)

    # 처음부터 훈력을 시작하지 않고 Checkpoint부터 시작하는 경우입니다.
    # 주요 파라미터들과 Vocab을 Checkpoint에서 가져옵니다.
    if args.resume:
        model_data = torch.load(args.resume)

        source_vocab = model_data['source_vocab']
        source_vocab_size = len(source_vocab)

        target_vocab = model_data['target_vocab']
        target_vocab_size = len(target_vocab)

        model = MorphemeRestoreModel(
            num_encoder_layers=model_data['num_encoder_layers'],
            num_decoder_layers=model_data['num_decoder_layers'],
            embedding_dim=model_data['embedding_dim'],
            source_vocab_size=model_data['source_vocab_size'],
            target_vocab_size=model_data['target_vocab_size'],
            num_heads=model_data['num_heads'],
            feedforward_dim=model_data['feedforward_dim'])

        model = model.to(DEVICE)
        model.load_state_dict(model_data['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        optimizer.load_state_dict(model_data['optimizer_state_dict'])

        start_epoch = model_data['epoch'] + 1

        train_dataset = RestoreDataset(args.train, args.max_characters, source_vocab=source_vocab, target_vocab=target_vocab)
    # 처음부터 훈련을 시작하는 경우입니다.
    else:
        train_dataset = RestoreDataset(args.train, args.max_characters)

        source_vocab = train_dataset.source_vocab
        source_vocab_size = len(source_vocab)

        target_vocab = train_dataset.target_vocab
        target_vocab_size = len(target_vocab)

        model = MorphemeRestoreModel(
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            embedding_dim=args.embedding_dim,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            num_heads=args.num_heads,
            feedforward_dim=args.feedforward_dim)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        start_epoch = 1

    # Padding을 하기 위해서 collate_fn에 make_batch 함수를 지정해줍니다.
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, collate_fn=make_batch)

    for epoch in range(start_epoch, args.epoch + 1):
        begin_time = datetime.datetime.now()
        loss = train(epoch, model, train_dataloader, source_vocab, target_vocab, loss_fn, optimizer)
        end_time = datetime.datetime.now()

        writer.add_scalar("Loss/train", loss, epoch)

        print(f'Epoch {epoch:>2d},\tloss: {loss:>.10f},\tTook {str(end_time - begin_time)}')

        writer.flush()

    print('Saving the final model')
    torch.save({
        'source_vocab_size': len(source_vocab),
        'target_vocab_size': len(target_vocab),
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'feedforward_dim': args.feedforward_dim,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'model_state_dict': model.state_dict(),
        'source_vocab': source_vocab,
        'target_vocab': target_vocab
    }, os.path.join(args.output, 'model_final.pth'))
    print('Done')
    
