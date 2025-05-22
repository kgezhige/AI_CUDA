import torch
import torch.nn as nn
import torch.nn.functional as F

class attention(nn.Module):
    def __init__(self,embdim,dropout=0.1, mask = None):
        super(attention,self).__init__()
        self.embdim = embdim
        self.mask = mask
        # [batch, seqlen, embdim]
        self.q = nn.Linear(embdim,embdim)
        self.k = nn.Linear(embdim,embdim)
        self.v = nn.Linear(embdim,embdim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v= self.v(x)
        scores = torch.matmul(q,k.transpose(-2,-1))
        scale = torch.sqrt(torch.tensor(self.embdim,dtype = torch.float32))
        scores = scores/scale
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, -1e9)

        atten_weights = F.softmax(scores,dim = -1)
        atten_weights = self.dropout(atten_weights)
        output = torch.matmul(atten_weights,v)
        return output,atten_weights
# 示例用法
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    
    # 创建随机输入数据
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 初始化模型
    model = attention(embdim=embed_dim, dropout=0.1)
    
    # 前向传播
    output, attn_weights = model(x)
    
    # 打印输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")