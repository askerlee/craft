import torch
from torch import nn, einsum
from einops import rearrange
import random

# max_pos_size = 160
class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width  = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        # rel_ind[i, j] = j - i + 159.
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        # q: [8, 1, 46, 62, 128]
        batch, heads, h, w, c = q.shape
        # self.rel_ind[:h, :h]: [46, 46]
        # self.rel_ind[:w, :w]: [62, 62]
        # rel_ind[i,j] = j - i + 159, precomputed distance between i, j. 
        # This assumes the input x (from which q is derived) is precisely on the grid.
        # This is fine when we do self-attention on x.
        # However, it will be somewhat limiting if we use RelPosEmb on cross-attention between two frames, 
        # particularly when we use flow_init != 0 (on sintel), 
        # we better get the positional encodings of x according to flow_init, instead of the grid of x.
        # However, an accurate computation of the relative distances between all input units is expensive.
        # Since values in flow_init are usually small, this inaccuracy may be negligible.
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb  = self.rel_width( self.rel_ind[:w, :w].reshape(-1))

        # height_emb: [46*46, 128] => [46, 46, 1,  128]
        # width_emb:  [62*62, 128] => [62, 1,  62, 128]
        # height_emb[i, j]: the embedding of element at (i,j) as a function of the height difference (i-j).
        # width_emb[i, j]:  the embedding of element at (i,j) as a function of the width  difference (i-j).
        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb  = rearrange(width_emb,  '(y v) d -> y () v d', y=w)
        
        # outer product? y, uv -> y u v              b  h  x   y   d        x  u   v   d
        # height_score: [8, 1, 46, 62, 46, 1]    <= [8, 1, 46, 62, 128] * [46, 46, 1,  128]
        # width_score:  [8, 1, 46, 62, 1,  62]
        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score  = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)
        # height_score + width_score: [8, 1, 46, 62, 46, 62], 65071232 elements.
        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)
        self.perturb_pos_embed_weight       = args.perturb_pos_embed_weight
        self.pos_embed_weight               = 1.0
        self.perturb_pos_embed_weight_range = self.pos_embed_weight * 0.2
        if args.position_and_content and self.perturb_pos_embed_weight and self.training:
            print("Positional embedding weight perturbation: {:.3}".format(self.perturb_pos_embed_weight_range))
        
    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        if self.args.position_only:
            sim = self.pos_emb(q)

        elif self.args.position_and_content:
            sim_content = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
            sim_pos = self.pos_emb(q)
            
            if self.perturb_pos_embed_weight and self.training:
                pew_noise = random.uniform(-self.perturb_pos_embed_weight_range, 
                                            self.perturb_pos_embed_weight_range)
            else:
                pew_noise = 0
                
            sim = sim_content + (self.pos_embed_weight + pew_noise) * sim_pos

        else:
            sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        return attn

# Aggregate output is dim-dimensional, same as the input. No FFN is used.
class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


if __name__ == "__main__":
    att = Attention(dim=128, heads=1)
    fmap = torch.randn(2, 128, 40, 90)
    out = att(fmap)

    print(out.shape)
