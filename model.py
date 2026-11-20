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
        self.register_buffer('tril_t',torch.tril(torch.ones((block_size,block_size))))

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

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.up=nn.Linear(n_embd,4*n_embd)
        self.down=nn.Linear(4*n_embd,n_embd)

    def forward(self,hidden_t):
        return self.down(torch.relu(self.up(hidden_t)))

class Block(nn.Module):
    def __init__(self,n_embd,n_head,block_size):
        super().__init__()
        self.attn=MultiHeadAttention(n_embd,n_head,block_size)
        self.ffn=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,hidden_t):
        hidden_t=hidden_t+self.attn(self.ln1(hidden_t))
        hidden_t=hidden_t+self.ffn(self.ln2(hidden_t))
        return hidden_t

class GPT(nn.Module):
    def __init__(self,vocab_size,n_embd,n_head,n_layer,block_size):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,n_embd)
        self.pos_emb=nn.Embedding(block_size,n_embd)
        self.blocks=nn.ModuleList([Block(n_embd,n_head,block_size)for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)

    def forward(self,inputs_t):
        T=inputs_t.shape[1]
        hidden_t=self.tok_emb(inputs_t)+self.pos_emb(torch.arange(T,device=inputs_t.device))
        for block in self.blocks:
            hidden_t=block(hidden_t)
        return self.lm_head(self.ln_f(hidden_t))