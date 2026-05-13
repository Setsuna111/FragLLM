from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim: int, dropout: float, ff_expansion: float = 2.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, int(ff_expansion * emb_dim)),
            nn.GELU(),
            nn.Linear(int(ff_expansion * emb_dim), emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.ffn(hidden_states)


class AttentionLayer(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, ff_expansion: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=ff_expansion)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residuals = hidden_states
        hidden_states = hidden_states.unsqueeze(0)
        attended, _ = self.attn(hidden_states, hidden_states, hidden_states)
        attended = attended.squeeze(0)
        hidden_states = self.ffn(residuals + attended) + residuals
        return self.output_layer_norm(hidden_states)


class CrossAttentionLayer(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=0.5)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, latents: Tensor, memory: Tensor) -> Tensor:
        residuals = latents
        latents_b = latents.unsqueeze(0)
        memory_b = memory.unsqueeze(0)
        attended, _ = self.attn(latents_b, memory_b, memory_b)
        attended = attended.squeeze(0)
        latents = self.ffn(residuals + attended) + residuals
        return self.output_layer_norm(latents)


class RelativePositionEncoder(nn.Module):
    def __init__(self, emb_dim: int, max_distance_bucket: int = 10, ratio_buckets: int = 8) -> None:
        super().__init__()
        self.distance_thresholds = (1, 2, 3, 4, 8, 16, 32, 64, 128)
        self.length_thresholds = (1, 4, 8, 16, 32, 64, 128, 256, 512)
        self.max_distance_bucket = max_distance_bucket
        self.ratio_buckets = ratio_buckets

        self.rel_start_emb = nn.Embedding(max_distance_bucket, emb_dim)
        self.rel_end_emb = nn.Embedding(max_distance_bucket, emb_dim)
        self.distance_emb = nn.Embedding(max_distance_bucket, emb_dim)
        self.length_emb = nn.Embedding(max_distance_bucket, emb_dim)
        self.ratio_emb = nn.Embedding(ratio_buckets, emb_dim)
        self.direction_emb = nn.Embedding(3, emb_dim)
        self.overlap_emb = nn.Embedding(2, emb_dim)
        self.type_emb = nn.Embedding(3, emb_dim)
        self.position_score_bias = nn.Embedding(max_distance_bucket, 1)
        self.direction_score_bias = nn.Embedding(3, 1)

    def _bucketize(self, values: Tensor, thresholds: Tuple[int, ...]) -> Tensor:
        values = values.abs().long()
        bucket = torch.zeros_like(values)
        for threshold in thresholds:
            bucket = bucket + (values >= threshold).long()
        return bucket.clamp(max=self.max_distance_bucket - 1)

    def _length_bucket(self, length: int, device: torch.device) -> Tensor:
        length_tensor = torch.tensor([length], device=device)
        return self._bucketize(length_tensor, self.length_thresholds)[0]

    def _ratio_bucket(self, numerator: Tensor, denominator: int) -> Tensor:
        if denominator <= 1:
            return torch.zeros_like(numerator, dtype=torch.long)
        ratio = numerator.float() / float(max(denominator - 1, 1))
        return torch.clamp((ratio * self.ratio_buckets).long(), max=self.ratio_buckets - 1)

    def local(self, positions: Tensor, start: int, end: int) -> Tensor:
        frag_len = max(end - start, 1)
        rel_start = positions - start
        rel_end = end - 1 - positions
        ratio_bucket = self._ratio_bucket(rel_start.clamp(min=0), frag_len)
        len_bucket = self._length_bucket(frag_len, positions.device)
        type_ids = torch.zeros_like(positions, dtype=torch.long)
        return (
            self.rel_start_emb(self._bucketize(rel_start, self.distance_thresholds))
            + self.rel_end_emb(self._bucketize(rel_end, self.distance_thresholds))
            + self.ratio_emb(ratio_bucket)
            + self.length_emb(len_bucket).unsqueeze(0)
            + self.type_emb(type_ids)
        )

    def sub_fragment(self, block_starts: Tensor, block_ends: Tensor, frag_start: int, frag_end: int) -> Tensor:
        centers = (block_starts + block_ends - 1) // 2
        frag_len = max(frag_end - frag_start, 1)
        rel_start = centers - frag_start
        rel_end = frag_end - 1 - centers
        ratio_bucket = self._ratio_bucket(rel_start.clamp(min=0), frag_len)
        len_bucket = self._length_bucket(frag_len, centers.device)
        type_ids = torch.ones_like(centers, dtype=torch.long)
        return (
            self.rel_start_emb(self._bucketize(rel_start, self.distance_thresholds))
            + self.rel_end_emb(self._bucketize(rel_end, self.distance_thresholds))
            + self.ratio_emb(ratio_bucket)
            + self.length_emb(len_bucket).unsqueeze(0)
            + self.type_emb(type_ids)
        )

    def global_context(
        self,
        block_starts: Tensor,
        block_ends: Tensor,
        frag_start: int,
        frag_end: int,
        total_len: int,
    ) -> Tensor:
        centers = (block_starts + block_ends - 1) // 2
        frag_center = (frag_start + frag_end - 1) // 2
        overlaps = (block_starts < frag_end) & (block_ends > frag_start)
        distance = torch.where(
            overlaps,
            torch.zeros_like(centers),
            torch.minimum((block_ends - frag_start).abs(), (block_starts - frag_end).abs()),
        )
        direction = torch.full_like(centers, 1, dtype=torch.long)
        direction = torch.where(centers < frag_center, torch.zeros_like(direction), direction)
        direction = torch.where(centers > frag_center, torch.full_like(direction, 2), direction)
        abs_ratio_bucket = self._ratio_bucket(centers.clamp(min=0), max(total_len, 1))
        type_ids = torch.full_like(centers, 2, dtype=torch.long)
        return (
            self.distance_emb(self._bucketize(distance, self.distance_thresholds))
            + self.direction_emb(direction)
            + self.overlap_emb(overlaps.long())
            + self.ratio_emb(abs_ratio_bucket)
            + self.type_emb(type_ids)
        )

    def score_bias(
        self,
        block_starts: Tensor,
        block_ends: Tensor,
        frag_start: int,
        frag_end: int,
    ) -> Tensor:
        centers = (block_starts + block_ends - 1) // 2
        frag_center = (frag_start + frag_end - 1) // 2
        overlaps = (block_starts < frag_end) & (block_ends > frag_start)
        distance = torch.where(
            overlaps,
            torch.zeros_like(centers),
            torch.minimum((block_ends - frag_start).abs(), (block_starts - frag_end).abs()),
        )
        direction = torch.full_like(centers, 1, dtype=torch.long)
        direction = torch.where(centers < frag_center, torch.zeros_like(direction), direction)
        direction = torch.where(centers > frag_center, torch.full_like(direction, 2), direction)
        distance_bucket = self._bucketize(distance, self.distance_thresholds)
        return (
            self.position_score_bias(distance_bucket).squeeze(-1)
            + self.direction_score_bias(direction).squeeze(-1)
        )


class BlockCompressor(nn.Module):
    def __init__(self, emb_dim: int, block_size: int, max_tokens: Optional[int] = None) -> None:
        super().__init__()
        self.block_size = max(block_size, 1)
        self.max_tokens = max_tokens
        self.score = nn.Linear(emb_dim, 1)
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, tokens: Tensor, positions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        seq_len = tokens.size(0)
        if seq_len == 0:
            raise ValueError("BlockCompressor received an empty token sequence")

        block_size = self.block_size
        if self.max_tokens is not None and self.max_tokens > 0:
            block_size = max(block_size, (seq_len + self.max_tokens - 1) // self.max_tokens)

        compressed = []
        block_starts = []
        block_ends = []
        for left in range(0, seq_len, block_size):
            right = min(left + block_size, seq_len)
            block = tokens[left:right]
            weights = torch.softmax(self.score(block).squeeze(-1), dim=0)
            compressed.append((weights.unsqueeze(-1) * block).sum(dim=0))
            block_starts.append(positions[left])
            block_ends.append(positions[right - 1] + 1)

        return (
            self.out_norm(torch.stack(compressed, dim=0)),
            torch.stack(block_starts, dim=0),
            torch.stack(block_ends, dim=0),
        )


class GlobalTopKSelector(nn.Module):
    def __init__(self, emb_dim: int, topk: int) -> None:
        super().__init__()
        self.topk = topk
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = emb_dim ** -0.5

    def forward(
        self,
        fragment_summary: Tensor,
        global_tokens: Tensor,
        block_starts: Tensor,
        block_ends: Tensor,
        frag_start: int,
        frag_end: int,
        pos_encoder: RelativePositionEncoder,
    ) -> Tensor:
        if global_tokens.size(0) == 0 or self.topk <= 0:
            return global_tokens[:0]

        query = self.query_proj(fragment_summary)
        keys = self.key_proj(global_tokens)
        scores = torch.matmul(keys, query) * self.scale
        scores = scores + pos_encoder.score_bias(block_starts, block_ends, frag_start, frag_end)

        overlaps = (block_starts < frag_end) & (block_ends > frag_start)
        if (~overlaps).any():
            scores = scores.masked_fill(overlaps, torch.finfo(scores.dtype).min)

        k = min(self.topk, global_tokens.size(0))
        topk_indices = torch.topk(scores, k=k, dim=0).indices
        topk_indices = topk_indices.sort().values
        return global_tokens[topk_indices]


class HierarchicalFragmentAdapter(nn.Module):
    def __init__(
        self,
        protein_emb_dim: int,
        text_emb_dim: int,
        latent_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        fragment_block_size: int = 4,
        global_block_size: int = 32,
        global_topk: int = 8,
        max_sub_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_size, protein_emb_dim))
        self.latent_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.position_encoder = RelativePositionEncoder(protein_emb_dim)
        self.fragment_compressor = BlockCompressor(
            protein_emb_dim, block_size=fragment_block_size, max_tokens=max_sub_tokens
        )
        self.global_compressor = BlockCompressor(
            protein_emb_dim, block_size=global_block_size, max_tokens=None
        )
        self.global_selector = GlobalTopKSelector(protein_emb_dim, topk=global_topk)
        self.cross_attention = CrossAttentionLayer(protein_emb_dim, num_heads, dropout)
        self.self_attention_layers = nn.ModuleList(
            [AttentionLayer(protein_emb_dim, num_heads, dropout, ff_expansion=1) for _ in range(num_layers - 1)]
        )
        self.output_proj = nn.Linear(protein_emb_dim, text_emb_dim, bias=False)
        self.out_layer_norm = nn.LayerNorm(text_emb_dim)

    def forward(self, protein_features: Tensor, position_ref: Tuple[int, int]) -> Tensor:
        total_len = protein_features.size(0)
        if total_len == 0:
            raise ValueError("HierarchicalFragmentAdapter received empty protein features")

        start = max(int(position_ref[0]), 0)
        end = min(int(position_ref[1]), total_len)
        if end <= start:
            end = min(start + 1, total_len)

        positions = torch.arange(total_len, device=protein_features.device)
        local_positions = positions[start:end]
        local_raw = protein_features[start:end]
        local_memory = local_raw + self.position_encoder.local(local_positions, start, end)

        sub_tokens, sub_starts, sub_ends = self.fragment_compressor(local_raw, local_positions)
        sub_memory = sub_tokens + self.position_encoder.sub_fragment(sub_starts, sub_ends, start, end)

        global_tokens, global_starts, global_ends = self.global_compressor(protein_features, positions)
        global_memory = global_tokens + self.position_encoder.global_context(
            global_starts, global_ends, start, end, total_len
        )

        fragment_summary = sub_memory.mean(dim=0)
        selected_global = self.global_selector(
            fragment_summary,
            global_memory,
            global_starts,
            global_ends,
            start,
            end,
            self.position_encoder,
        )

        memory = torch.cat([local_memory, sub_memory, selected_global], dim=0)
        latents = self.latent_layer_norm(self.latents)
        latents = self.cross_attention(latents, memory)
        for layer in self.self_attention_layers:
            latents = layer(latents)
        return self.out_layer_norm(self.output_proj(latents))
