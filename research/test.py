import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

import argparse
import math

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

UNK_INDEX = 0
PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

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

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

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
        
# 매 단계에서 가장 확률이 높은 토큰을 취합니다. 대체 방법으로는 beam search 같은 방법도 있습니다.
def greedy_decode(model, source, source_mask, max_len, start_symbol):
    source = source.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    memory = model.encode(source, source_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        # 미래의 토큰은 보지 않도록 Mask를 해줍니다.
        target_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        # Greedy하게 각 단계에서 가장 높은 토큰을 취합니다.
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # autoregressive 하도록 지금까지 생성된 토큰을 다음 입력으로 계속 붙여줍니다.
        ys = torch.cat([ys, torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)

        # <eos>를 만나면 생성을 멈춥니다.
        if next_word == EOS_INDEX:
            break

    return ys

def restore_morphemes(model, source_vocab, target_vocab, source):
    model.eval()

    # 훈련 때처럼 문장의 앞 뒤에 <bos>, <eos>를 붙여줍니다.
    tokens = [BOS_INDEX] + [source_vocab[token] for token in source] + [EOS_INDEX]
    num_tokens = len(tokens)
    source = torch.LongTensor(tokens).reshape(num_tokens, 1)
    source_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    # 출력을 최대 300 토큰으로 제한하고 있습니다.
    target_tokens = greedy_decode(model, source, source_mask, max_len=300, start_symbol=BOS_INDEX).flatten()

    return ''.join([target_vocab.get_itos()[token] for token in target_tokens]).replace('<bos>', '').replace('<eos>', '')

if __name__ == '__main__':
    print(f'Using {DEVICE}')

    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--model')

    args = parser.parse_args()

    model_data = torch.load(args.model)

    source_vocab_size = model_data['source_vocab_size']
    source_vocab = model_data['source_vocab']
    target_vocab_size = model_data['target_vocab_size']
    target_vocab = model_data['target_vocab']

    model = MorphemeRestoreModel(
            num_encoder_layers=model_data['num_encoder_layers'],
            num_decoder_layers=model_data['num_decoder_layers'],
            embedding_dim=model_data['embedding_dim'],
            source_vocab_size=model_data['source_vocab_size'],
            target_vocab_size=model_data['target_vocab_size'],
            num_heads=model_data['num_heads'],
            feedforward_dim=model_data['feedforward_dim'])

    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(DEVICE)

    model.eval()

    with torch.no_grad():
        while True:
            source = input('Description: ')
            if source:
                print('Answer:', restore_morphemes(model, source_vocab, target_vocab, source))
            else:
                break
                
