import torch
torch.manual_seed(1337)
from model import Head,MultiHeadAttention

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