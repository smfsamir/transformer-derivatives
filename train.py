from tqdm import tqdm
from torchinfo import summary
import copy
import math
from torch.optim.lr_scheduler import LambdaLR
import os
import ipdb
import loguru
from typing import Dict
import torch.nn as nn
import torch
from torch.utils.data import BatchSampler, RandomSampler, Dataset, DataLoader
from collections import OrderedDict
from torch.nn.functional import log_softmax

from flowmason import conduct, SingletonStep, MapReduceStep
from dotenv import dotenv_values

DEVICE = dotenv_values(".env")['DEVICE']

logger = loguru.logger
class DerivativeDataset(Dataset):
    def __init__(self, file_path): 
        self.data = [line.strip() for line in open(file_path, 'r').readlines()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def step_create_token_dict(**kwargs) -> Dict[str, int]:
    unique_tokens = set()
    with open('train.txt', 'r') as f:
        for derivative_line in f:
            unique_tokens.update(derivative_line.strip())
    # add [PAD], [BOS], [EOS] to the unique tokens
    unique_tokens.add('[PAD]')
    unique_tokens.add('[BOS]')
    unique_tokens.add('[EOS]')
    return {token: i for i, token in enumerate(unique_tokens)}

def collate_batch(batch, tokenizer_dict, device):
    bos_id = tokenizer_dict['[BOS]']
    eos_id = tokenizer_dict['[EOS]']

    src = [datapoint.split('=')[0] for datapoint in batch]
    tgt = [datapoint.split('=')[1] for datapoint in batch]
    src = [[bos_id] + [tokenizer_dict[token] for token in src] + [eos_id] for src in src]
    tgt = [[bos_id] + [tokenizer_dict[token] for token in tgt] + [eos_id] for tgt in tgt]

    # pad the sequences with [PAD] token
    max_src_len = max([len(src) for src in src])
    max_tgt_len = max([len(tgt) for tgt in tgt])

    src = [src + [tokenizer_dict['[PAD]']] * (max_src_len - len(src)) for src in src]
    tgt = [tgt + [tokenizer_dict['[PAD]']] * (max_tgt_len - len(tgt)) for tgt in tgt]
    return torch.tensor(src).to(device), torch.tensor(tgt).to(device)

def create_dataloaders(
    device,
    tokenizer_dict,
    batch_size=256,
) -> DataLoader:
    
    dataset = DerivativeDataset('train.txt')

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenizer_dict,
            device
        )


    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_dataloader

# model = nn.Transformer(
#     n_head = 4, 
# )
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

        
class DerivativeTransformer(nn.Module):

    def __init__(self, transformer, src_embed, tgt_embed, generator):
        super(DerivativeTransformer, self).__init__()
        self.transformer = transformer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_attn_mask, tgt_attn_mask,
                    src_pad_mask, tgt_pad_mask):
        src_encoded = self.src_embed(src)
        tgt_encoded = self.tgt_embed(tgt)
        # TODO: tgt attn mask should be True/False, not -inf
        out = self.transformer(
            src_encoded, tgt_encoded, src_mask=src_attn_mask, tgt_mask=tgt_attn_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask
        )
        # assert out.shape == tgt.shape
        return self.generator(out) # NOTE: need to check the shape of this.


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe) # wtf is this?

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # return self.dropout(x)
        return x

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

def make_model(tokenizer_dict, device):
    c = copy.deepcopy
    d_model = 512
    position = PositionalEncoding(d_model)
    embed_src = nn.Sequential(Embeddings(d_model, len(tokenizer_dict), padding_idx=tokenizer_dict['[PAD]']), c(position))
    embed_tgt = nn.Sequential(Embeddings(d_model, len(tokenizer_dict), padding_idx=tokenizer_dict['[PAD]']), c(position))
    generator = Generator(d_model, len(tokenizer_dict))
    model = DerivativeTransformer(
        nn.Transformer(d_model=d_model, 
                       nhead=2, 
                       num_encoder_layers=3, 
                       num_decoder_layers=3, 
                       dim_feedforward=2048),
        embed_src,
        embed_tgt,
        generator
    )
    return model.to(device)

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def step_train_model(tokenizer_dict, **kwargs):
    batch_size = 256
    dataloader = create_dataloaders(
        device=torch.device(DEVICE),
        tokenizer_dict=tokenizer_dict,
        batch_size=batch_size
    )
    i2t = {v: k for k, v in tokenizer_dict.items()} 
    model = make_model(tokenizer_dict, DEVICE)
    summary(model)
    ipdb.set_trace()
    loss = nn.CrossEntropyLoss(ignore_index=tokenizer_dict['[PAD]'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=4000
        ),
    )
    model.train()
    eval_steps = 500
    curr_step = 0
    best_loss = float('inf')


    for src, tgt in dataloader: # B x S
        src = src.T # batch first = False
        tgt = tgt.T # batch first = False

        # NOTE: need to replace with -100 maybe?
        src_padding_mask = (src == tokenizer_dict['[PAD]'])
        tgt_padding_mask = (tgt == tokenizer_dict['[PAD]'])
        src_attn_mask = src == tokenizer_dict['[PAD]']
        num_heads = model.transformer.nhead

        # (N * num_heads, S, S) for the attn mask
        # thus, we need to repeat the mask for the number of heads using tile


        # TODO; do we need a src attn mask? maybe not
        tgt_attn_mask = model.transformer.generate_square_subsequent_mask(tgt.size(0)).unsqueeze(0).repeat(src.shape[1] * num_heads, 1, 1)
        # turn -inf to False and 0 to True
        tgt_attn_mask = tgt_attn_mask != 0

        # (S, N, E)  
        src_attn_mask = None
        logits = model(src, tgt, src_attn_mask, 
                       tgt_attn_mask.to(DEVICE), src_padding_mask.T.to(DEVICE), tgt_padding_mask.T.to(DEVICE))
        # transpose the logits to be B x S x V
        loss_output = loss(logits.permute(1, 2, 0)[:, :, :-1], tgt.T[:, 1:])
        # loss_output = loss(logits.permute(1, 0, 2)[:, :-1, :], tgt.T[:, 1:])
        logger.info(f'Loss: {loss_output.item()} (curr_step: {curr_step})')

        # update the model
        loss_output.backward()
        optimizer.step()
        lr_scheduler.step()
        # zero the gradients
        optimizer.zero_grad()

        if curr_step > 0 and curr_step % eval_steps == 0:
            test_seq = "d(8exp^(9e))/de=72exp^(9e)"
            model.eval()
            with torch.no_grad():
                eval_src = collate_batch([test_seq], tokenizer_dict, torch.device(DEVICE))[0].T
                eval_tgt = torch.tensor([tokenizer_dict['[BOS]']]).unsqueeze(0).T.to(DEVICE)
                next_token = None
                curr_seq = ""
                while next_token != '[EOS]' and len(curr_seq) < 30:
                    assert tokenizer_dict['[PAD]'] not in eval_src
                    logits = model(eval_src, eval_tgt, None, None, None, None)
                    # get the next token
                    next_token_logits = logits[-1, 0, :]
                    next_token = torch.argmax(next_token_logits)
                    # log the next token by getting it from the tokenizer using i2t
                    next_token = i2t[next_token.item()]
                    curr_seq += next_token
                    # add the next token to the eval_tgt
                    eval_tgt = torch.cat([eval_tgt, torch.tensor([[tokenizer_dict[next_token]]]).T.to(DEVICE)], dim=0)
                logger.info(f'Predicted sequence: {curr_seq}')
            # save the current model
            if loss_output < best_loss:
                best_loss = loss_output
                saved_model_name = f"best_model_{curr_step}_enc={num_encoder_layers}_dec={num_decoder_layers}_nheads={num_attn_heads}_bs={batch_size}.pt"
                torch.save(model.state_dict(), saved_model_name)
            model.train()
            
        curr_step += 1
    return saved_model_name

def step_eval_model(tokenizer_dict, eval_fname: str, model_name: str, 
                    **kwargs): 
    # load the model
    model = make_model(tokenizer_dict, DEVICE)
    summary(model)
    ipdb.set_trace()
    model.load_state_dict(torch.load(model_name))
    batch_size = 256
    dataloader = create_dataloaders(
        device=torch.device(DEVICE),
        tokenizer_dict=tokenizer_dict,
        batch_size=batch_size
    )
    model.eval()

    num_correct = 0
    num_total = 0
    max_seq_len = 30
    summary(model) # print the model summary
    curr_batch = 0
    with torch.no_grad():
        i2t = {v: k for k, v in tokenizer_dict.items()} 
        for src, tgt in tqdm(dataloader):
            src = src.T
            tgt = tgt.T
            decoder_prefix = tgt[0].unsqueeze(0)
            # set while condition to while we decoder prefix is less than max_seq_len + 2
            while decoder_prefix.shape[0] < max_seq_len + 2:
                src_padding_mask = (src == tokenizer_dict['[PAD]'])
                logits = model(src, decoder_prefix, None, None, src_padding_mask.T, None)
                next_token_logits = logits.permute(1, 0, 2)[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=1)
                decoder_prefix = torch.cat([decoder_prefix, next_tokens.unsqueeze(0)], dim=0)
            predictions = decoder_prefix.T
            # assert that tokenizer_dict['[EOS]'] is in all of the predictions
            assert (predictions == tokenizer_dict['[EOS]']).sum(axis=1).sum() >= batch_size
            # take the prediction up to the first [EOS] token for each datapoint
            predictions_decoded = []
            actual_decoded = []
            for i in range(8):
                pred = predictions[i]
                actual = tgt.T[i]
                for j in range(len(pred)):
                    if pred[j] == tokenizer_dict['[EOS]']:
                        pred = pred[:j]
                        break
                for j in range(len(actual)):
                    if actual[j] == tokenizer_dict['[EOS]']:
                        actual = actual[:j]
                        break
                pred = [i2t[token.item()] for token in pred]
                actual = [i2t[token.item()] for token in actual]
                # remove [BOS] and [EOS] tokens
                pred = pred[1:]
                actual = actual[1:]
                predictions_decoded.append(''.join(pred))
                actual_decoded.append(''.join(actual))
                num_correct += int(''.join(pred) == ''.join(actual))
                num_total += 1
            curr_batch += 1
            if curr_batch % 10 == 0:
                logger.info(f'Accuracy: {num_correct / num_total}')
    logger.info(f'Accuracy: {num_correct / num_total}') 
    # return num_correct / num_total

if __name__ == '__main__':
    step_dict = OrderedDict()
    step_dict['create_tokenizer'] = SingletonStep(step_create_token_dict, {
        'version': '001'
    })
    step_dict['train_model'] = SingletonStep(step_train_model, {
        'tokenizer_dict': 'create_tokenizer',
        'version': '001' 
    })
    eval_dict = OrderedDict()
    eval_dict['eval_model'] = SingletonStep(step_eval_model, {
        'version': '001'
    })
    step_dict['eval_model_map'] = MapReduceStep(eval_dict, {
        'model_name': list(filter(lambda x: x.endswith('.pt'), os.listdir()))
    },{
        'tokenizer_dict': 'create_tokenizer', 
        'model_name': 'train_model',
        'version': '001', 
        'eval_fname': 'train.txt'
    },  list)
    conduct('cache_dir', step_dict, 'scale_logs')