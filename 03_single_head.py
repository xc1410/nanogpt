import torch
import torch.nn as nn
import torch.nn.functional as F

with open('data/input.txt',encoding='utf-8') as f:
    text=f.read()
chars=sorted(set(text))
vocab_size=len(chars)
stoi={}
for index,ch in enumerate(chars):
    stoi[ch]=index
ids=[stoi[ch] for ch in text]
split=len(ids)*9//10
train_ids_t=torch.tensor(ids[:split],dtype=torch.int64)
val_ids_t=torch.tensor(ids[split:],dtype=torch.int64)
print(len(train_ids_t),len(val_ids_t))
print(train_ids_t[:10])
torch.manual_seed(1337)
block_size=8
batch_size=4

def get_batch(data_t):
    starts_t=torch.randint(len(data_t)-block_size,(batch_size,))
    inputs_t=torch.stack([data_t[start:start+block_size]for start in starts_t])
    targets_t=torch.stack([data_t[start+1:start+block_size+1]for start in starts_t])
    return inputs_t,targets_t

inputs_t,targets_t=get_batch(train_ids_t)
print(inputs_t.shape,targets_t.shape)
print(inputs_t)
print(targets_t)
assert inputs_t.shape==(batch_size,block_size),inputs_t.shape
assert targets_t.shape==(batch_size,block_size),targets_t.shape
assert (inputs_t[:,1:]==targets_t[:,:-1]).all()

n_emb=32
tok_emb=nn.Embedding(vocab_size,n_emb)
tok_vectors_t=tok_emb(inputs_t)
print(tok_emb.weight.shape,tok_vectors_t.shape)
assert tok_vectors_t.shape==(batch_size,block_size,n_emb),tok_vectors_t.shape
onehot_t=F.one_hot(inputs_t,num_classes=vocab_size).to(torch.float)
assert torch.allclose(onehot_t@tok_emb.weight,tok_vectors_t)
print(tok_vectors_t[0,0,:4])
print(tok_emb.weight[24,:4])