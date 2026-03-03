import torch
import torch.nn as nn

from argparse import Namespace


class VectorQuantizer(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self._init_resources()

    def _init_resources(self):
        config = self.config
        num_embeddings = config.quantizer.get("num_embeddings")
        embedding_dim = config.quantizer.get("embedding_dim")
        self.embedding = nn.Embedding(
                num_embeddings,
                embedding_dim
            )
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, X, return_indices=False):
        z_e = X.permute(0, 2, 3, 1).contiguous()
        z_e_flattened = z_e.view(-1, self.embedding.weight.shape[-1])

        d = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) +\
            torch.sum(self.embedding.weight ** 2, dim = 1) - 2 *\
            torch.matmul(z_e_flattened, self.embedding.weight.t())

        min_embedding_idxs = torch.argmin(d, dim=1).unsqueeze(1)
        min_embedding_one_hot_vectors = torch.zeros(
                *d.shape
                ).to(z_e.device)
        min_embedding_one_hot_vectors.scatter_(1, min_embedding_idxs, 1)

        z_q = torch.matmul(min_embedding_one_hot_vectors, self.embedding.weight).view(z_e.shape)
        #z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_embedding_one_hot_vectors
