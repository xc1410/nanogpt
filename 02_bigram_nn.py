import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

with open('data/input.txt',encoding='utf-8') as f:
    text=f.read()
chars=sorted(set(text))
vocab_size=len(chars)
stoi = {}
for index,ch in enumerate(chars):
    stoi[ch]=index
ids=[stoi[ch] for ch in text]
split=len(ids)*9//10
train_ids=ids[:split]
val_ids=ids[split:]

class PairData(Dataset):
    def __init__(self,char_ids):
        super().__init__()
        self.inputs_t=torch.tensor(char_ids[:-1],dtype=torch.int64)
        self.labels_t=torch.tensor(char_ids[1:],dtype=torch.int64)

    def __len__(self):
        return len(self.labels_t)

    def __getitem__(self,index):
        return self.inputs_t[index],self.labels_t[index]

class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.table=nn.Parameter(torch.zeros((vocab_size,vocab_size)))

    def forward(self,inputs_t):
        onehot_t=F.one_hot(inputs_t,num_classes=self.vocab_size).to(torch.float32)
        return onehot_t@self.table

lr=20
batch_size=1000
epochs=10
train_data=PairData(train_ids)
val_data=PairData(val_ids)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
print(len(train_data),len(train_loader),len(val_data))
model=Bigram(vocab_size)
optimizer=torch.optim.SGD(model.parameters(),lr=lr)
logits_t=model(val_data.inputs_t)
print(F.cross_entropy(logits_t,val_data.labels_t).item())

begin=time.perf_counter()
for epoch in range(epochs):
    epoch_loss=0
    for inputs_t,labels_t in train_loader:
        optimizer.zero_grad()
        logits_t=model(inputs_t)
        assert logits_t.shape==(len(labels_t),vocab_size),logits_t.shape
        loss_t=F.cross_entropy(logits_t,labels_t)
        loss_t.backward()
        optimizer.step()
        epoch_loss=epoch_loss+loss_t.item()
    logits_t=model(val_data.inputs_t)
    val_loss=F.cross_entropy(logits_t,val_data.labels_t).item()
    print(epoch,epoch_loss/len(train_loader),val_loss)
seconds = time.perf_counter() - begin
print(seconds, len(train_data) * epochs / seconds)
assert val_loss < 2.6, val_loss
    