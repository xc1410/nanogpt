import torch
torch.manual_seed(1337)
from model import Head,MultiHeadAttention,FeedForward,Block,GPT

head=Head(32,16,8)
print(head)
print(head.tril_t.shape)
print(len(list(head.parameters())))

embedded_t=torch.randn((4,8,32))
out_t=head(embedded_t)
assert out_t.shape==(4,8,16),out_t.shape

print(out_t[0,7,:3])
assert torch.allclose(head.attn_weights_t.sum(dim=2),torch.ones((4,8)))
assert (head.attn_weights_t*(head.tril_t==0)).sum()==0

mha=MultiHeadAttention(32,4,8)
print(mha)
print(len(list(mha.parameters())))

mha_out_t=mha(embedded_t)
assert mha_out_t.shape==(4,8,32),mha_out_t.shape
print(mha_out_t[0,7,:3])

ffn=FeedForward(32)
for param in ffn.parameters():
    print(param.shape)
ffn_out_t=ffn(mha_out_t)
assert ffn_out_t.shape==(4,8,32),ffn_out_t.shape
print(ffn_out_t[0,7,:3])
block=Block(32,4,8)
print(block.ln1)
print(len(list(block.parameters())))

block_out_t=block(embedded_t)
assert block_out_t.shape==(4,8,32),block_out_t.shape
print(block_out_t[0,7,:3])

gpt=GPT(65,32,4,2,8)
print(gpt.tok_emb.weight.shape)
print(gpt.pos_emb.weight.shape)

n_params=sum(param.numel()for param in gpt.parameters())
print(n_params)
assert n_params==29761,n_params

inputs_t=torch.randint(65,(4,8))
logits_t=gpt(inputs_t)
assert logits_t.shape==(4,8,65),logits_t.shape
print(logits_t[0,7,:3])