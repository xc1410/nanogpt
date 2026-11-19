import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self,n_embd,head_size,block_size):
        super().__init__()
        self.head_size=head_size
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.tril_t=torch.tril(torch.ones((block_size,block_size)))

    def forward(self,embedded_t):
        T=embedded_t.shape[1]
        scores_t=self.query(embedded_t)@self.key(embedded_t).transpose(1,2)/self.head_size**0.5
        scores_t=scores_t.masked_fill(self.tril_t[:T,:T]==0,float('-inf'))
        attn_weights_t=F.softmax(scores_t,dim=2)
        self.attn_weights_t=attn_weights_t
        return attn_weights_t@self.value((embedded_t))
class MultiHeadAttention(nn.Module):
    def __init__(self,n_embd,n_head,block_size):
        super().__init__()
        head_size=n_embd//n_head
        self.heads=nn.ModuleList([Head(n_embd,head_size,block_size)for _ in range(n_head)])
        self.proj=nn.Linear(n_embd,n_embd)

    def forward(self,embedded_t):
        heads_out=torch.cat([head(embedded_t)for head in self.heads],dim=2)
        return self.proj(heads_out)