import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
import time
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
block_size=64
batch_size=4

def get_batch(data_t):
    starts_t=torch.randint(len(data_t)-block_size,(batch_size,))
    inputs_t=torch.stack([data_t[start:start+block_size]for start in starts_t])
    targets_t=torch.stack([data_t[start+1:start+block_size+1]for start in starts_t])
    return inputs_t.to(device),targets_t.to(device)

inputs_t,targets_t=get_batch(train_ids_t)
print(inputs_t.shape,targets_t.shape)
print(inputs_t)
print(targets_t)
assert inputs_t.shape==(batch_size,block_size),inputs_t.shape
assert targets_t.shape==(batch_size,block_size),targets_t.shape
assert (inputs_t[:,1:]==targets_t[:,:-1]).all()

model=GPT(vocab_size,128,4,4,block_size).to(device)
n_params=sum(p.numel()for p in model.parameters())
assert n_params==816705,n_params
logits_t=model(inputs_t)
loss_t=F.cross_entropy(logits_t.reshape(-1,vocab_size),targets_t.reshape(-1))
print(logits_t.shape,loss_t.item())
assert logits_t.shape==(batch_size,block_size,vocab_size),logits_t.shape

optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
optimizer.zero_grad()
loss_t.backward()
optimizer.step()
loss_after_t=F.cross_entropy(model(inputs_t).reshape(-1,vocab_size),targets_t.reshape(-1))
print(loss_t.item(),loss_after_t.item())

def estimate_loss(data_t):
    total=0.0
    with torch.no_grad():
        for _ in range(20):
            inputs_t,targets_t=get_batch(data_t)
            total+=F.cross_entropy(model(inputs_t).reshape(-1,vocab_size),targets_t.reshape(-1)).item()
    return total/20
      
batch_size=32
start=time.perf_counter()
for step in range(2000):
    inputs_t,targets_t=get_batch(train_ids_t)
    optimizer.zero_grad()
    loss_t=F.cross_entropy(model(inputs_t).reshape(-1,vocab_size),targets_t.reshape(-1))
    loss_t.backward()
    optimizer.step()
    if step%500==0 or step==1999:
        train_loss=estimate_loss(train_ids_t)
        val_loss=estimate_loss(val_ids_t)
        print(step,train_loss,val_loss)
print('训练用时',time.perf_counter()-start)
assert loss_t.item()<3.0,loss_t.item()
assert val_loss>1.0 and val_loss<1.80,val_loss

ids_t=torch.zeros((1,1),dtype=torch.int64,device=device)
logits_t=model(ids_t)
probs_t=F.softmax(logits_t[:,-1,:],dim=1)
next_t=torch.multinomial(probs_t,num_samples=1)
print(probs_t.shape,next_t,chars[next_t.item()])
with torch.no_grad():
    for _ in range(300):
        logits_t=model(ids_t[:,-block_size:])
        next_t=torch.multinomial(F.softmax(logits_t[:,-1,:],dim=1),num_samples=1)
        ids_t=torch.cat([ids_t,next_t],dim=1)
assert ids_t.shape==(1,301),ids_t.shape
print(''.join(chars[i] for i in ids_t[0]))