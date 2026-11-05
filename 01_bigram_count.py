import time
import numpy as np

with open('data/input.txt',encoding='utf-8') as f:
    text=f.read()
chars=sorted(set(text))
vocab_size=len(chars)
print(len(text),vocab_size)
assert vocab_size==65,vocab_size

stoi={}
for index,ch in enumerate(chars):
    stoi[ch]=index
ids=[stoi[ch] for ch in text]
split=len(ids)*9//10
train_ids=ids[:split]
val_ids = ids[split:]
print(len(train_ids),len(val_ids))

begin = time.perf_counter()
counts=np.zeros((vocab_size,vocab_size))
for prev_id,next_id in zip(train_ids,train_ids[1:]):
    counts[prev_id,next_id]=counts[prev_id,next_id]+1
seconds=time.perf_counter()-begin
print(seconds,counts.sum(),counts[stoi['q'],stoi['u']])
assert counts.sum()==len(train_ids)-1,counts.sum()

probs=(counts+1)/(counts+1).sum(axis=1,keepdims=True)
assert abs (probs[stoi['e']].sum()-1)<1e-9,probs[stoi['e']].sum()
print(probs[stoi['q'], stoi['u']])


train_nll = -np.log(probs[train_ids[:-1], train_ids[1:]]).mean()
val_nll = -np.log(probs[val_ids[:-1], val_ids[1:]]).mean()
print(train_nll, val_nll)


sample = ''
current_id = stoi['\n']
for step in range(300):
    current_id = np.random.choice(vocab_size, p=probs[current_id])
    sample = sample + chars[current_id]
print(sample)