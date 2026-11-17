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
pos_emb=nn.Embedding(block_size,n_emb)
positions_t=torch.arange(block_size)
print(positions_t)
pos_vectors_t=pos_emb(positions_t)
embedded_t=tok_vectors_t+pos_vectors_t
print(pos_vectors_t.shape,embedded_t.shape)
assert pos_vectors_t.shape==(block_size,n_emb),pos_vectors_t.shape
assert embedded_t.shape==(batch_size,block_size,n_emb),embedded_t.shape
assert torch.allclose(embedded_t[1,3],tok_vectors_t[1,3]+pos_vectors_t[3])
ones_t=torch.ones((block_size,block_size))
tril_t=torch.tril(ones_t)
print(tril_t)
avg_weights_t=tril_t/tril_t.sum(dim=1,keepdim=True)
print(avg_weights_t[:4,:4])
averaged_t=avg_weights_t@embedded_t
print(averaged_t.shape)
assert torch.allclose(avg_weights_t.sum(dim=1),torch.ones(block_size))
assert torch.allclose(averaged_t[0,2],embedded_t[0,:3].mean(dim=0))
head_size=16
query=nn.Linear(n_emb,head_size,bias=False)
key=nn.Linear(n_emb,head_size,bias=False)
queries_t=query(embedded_t)
keys_t=key(embedded_t)
print(queries_t.shape,keys_t.shape)
scores_t=queries_t@keys_t.transpose(1,2)/head_size**0.5
print(scores_t.shape)
print(scores_t[0,:3,:3])
assert scores_t.shape==(batch_size,block_size,block_size),scores_t.shape
assert torch.allclose(scores_t[0,2,1],(queries_t[0,2]*keys_t[0,1]).sum()/head_size**0.5)
masked_scores_t=scores_t.masked_fill(tril_t==0,float('-inf'))
print(masked_scores_t[0,:3,:3])
attn_weights_t=F.softmax(masked_scores_t,dim=2)
print(attn_weights_t[0,:3,:3])
assert torch.allclose(attn_weights_t.sum(dim=2),torch.ones((batch_size,block_size)))
assert (attn_weights_t*(tril_t==0)).sum()==0
value=nn.Linear(n_emb,head_size,bias=False)
values_t=value(embedded_t)
head_out_t=attn_weights_t@values_t
print(values_t.shape,head_out_t.shape)
assert head_out_t.shape==(batch_size,block_size,head_size),head_out_t.shape
assert torch.allclose(head_out_t[0,1],attn_weights_t[0,1,0]*values_t[0,0]+attn_weights_t[0,1,1]*values_t[0,1])
lm_head=nn.Linear(head_size,vocab_size)
logits_t=lm_head(head_out_t)
print(logits_t.shape)
loss_t=F.cross_entropy(logits_t.reshape(-1,vocab_size),targets_t.reshape(-1))
print(loss_t.item())
assert logits_t.shape==(batch_size,block_size,vocab_size),logits_t.shape

def forward(inputs_t):
    embedded_t=tok_emb(inputs_t)+pos_emb(positions_t)
    scores_t=query(embedded_t)@key(embedded_t).transpose(1,2)/head_size**0.5
    attn_weights_t=F.softmax(scores_t.masked_fill(tril_t==0,float('-inf')),dim=2)
    return lm_head(attn_weights_t@value(embedded_t))

assert torch.allclose(forward(inputs_t),logits_t)
params=[]
for layer in [tok_emb,pos_emb,query,key,value,lm_head]:
    params+=list(layer.parameters())
print(len(params))
optimizer=torch.optim.SGD(params,lr=0.5)
batch_size=32
for step in range(1000):
    inputs_t,targets_t=get_batch(train_ids_t)
    optimizer.zero_grad()
    loss_t=F.cross_entropy(forward(inputs_t).reshape(-1,vocab_size),targets_t.reshape(-1))
    loss_t.backward()
    optimizer.step()
    if step%200==0 or step==999:
        print(step,loss_t.item())
assert loss_t.item()<3.0,loss_t.item()