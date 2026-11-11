# -*- coding: utf-8 -*-
"""Inductive Heterogeneous Graph Neural Network (HR KG + GNN) using PyTorch Geometric.

Models a multi-relational HR Knowledge Graph with node types:
  job, skill, industry, location, role, exp, qual
and bidirectional relations plus skill co-occurrence (CO_OCCURS_WITH).

Supports:
- Self-supervised multi-relation link prediction training on data/jobs_dataset.csv
- Exact O(deg(u)) inductive inference for newly uploaded CVs (LinkSAGE-style)
- Fast checkpoint persistence for instant Flask serving
"""

import os
import random
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import (
    GNN_CHECKPOINT_PATH,
    GNN_EPOCHS,
    GNN_HIDDEN_DIM,
    GNN_LR,
    GNN_TEXT_DIM,
)
from text_processing import norm_text

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    HAS_PYG = True
except Exception:
    HAS_PYG = False


ENTITY_TYPES = ["skill", "industry", "location", "role", "exp", "qual"]

REL_SPECS = [
    ("job", "requires_skill", "skill"),
    ("skill", "rev_requires_skill", "job"),
    ("job", "belongs_to_industry", "industry"),
    ("industry", "rev_belongs_to_industry", "job"),
    ("job", "located_in", "location"),
    ("location", "rev_located_in", "job"),
    ("job", "has_role", "role"),
    ("role", "rev_has_role", "job"),
    ("job", "requires_exp", "exp"),
    ("exp", "rev_requires_exp", "job"),
    ("job", "requires_qual", "qual"),
    ("qual", "rev_requires_qual", "job"),
    ("skill", "co_occurs_with", "skill"),
]


if HAS_PYG:
    class HeteroHRGNN(nn.Module):
        """2-layer Heterogeneous GraphSAGE with residual projection for HR KG."""

        def __init__(self, text_dim: int, hidden_dim: int, entity_counts: dict):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.job_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.entity_embs = nn.ModuleDict({
                etype: nn.Embedding(max(1, count), hidden_dim)
                for etype, count in entity_counts.items()
            })

            self.conv1 = HeteroConv(
                {rel: SAGEConv((-1, -1), hidden_dim) for rel in REL_SPECS},
                aggr="mean",
            )
            self.norm1 = nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_dim)
                for ntype in ["job"] + list(entity_counts.keys())
            })

            self.conv2 = HeteroConv(
                {rel: SAGEConv((-1, -1), hidden_dim) for rel in REL_SPECS},
                aggr="mean",
            )
            self.norm2 = nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_dim)
                for ntype in ["job"] + list(entity_counts.keys())
            })
            self.dropout = nn.Dropout(0.15)

        def initial_features(self, job_x: torch.Tensor, device: torch.device) -> dict:
            x_dict = {"job": self.job_proj(job_x)}
            for etype, emb in self.entity_embs.items():
                idx = torch.arange(emb.num_embeddings, device=device)
                x_dict[etype] = emb(idx)
            return x_dict

        def forward_layer1(self, x_dict: dict, edge_index_dict: dict) -> dict:
            out = self.conv1(x_dict, edge_index_dict)
            h1 = {}
            for ntype, x in x_dict.items():
                if ntype in out:
                    h = self.norm1[ntype](out[ntype] + x)
                    h1[ntype] = self.dropout(F.relu(h))
                else:
                    h1[ntype] = x
            return h1

        def forward_layer2(self, h1_dict: dict, edge_index_dict: dict) -> dict:
            out = self.conv2(h1_dict, edge_index_dict)
            z = {}
            for ntype, h in h1_dict.items():
                if ntype in out:
                    val = self.norm2[ntype](out[ntype] + h)
                else:
                    val = h
                z[ntype] = F.normalize(val, p=2, dim=-1)
            return z

        def forward(self, job_x: torch.Tensor, edge_index_dict: dict) -> tuple[dict, dict]:
            x_dict = self.initial_features(job_x, job_x.device)
            h1_dict = self.forward_layer1(x_dict, edge_index_dict)
            z_dict = self.forward_layer2(h1_dict, edge_index_dict)
            return h1_dict, z_dict


def _extract_job_entities(job_info: dict, job_nodes: list, G=None):
    """Build contiguous entity vocabularies and edge lists from job_info and graph G."""
    vocabs = {etype: {"__UNK__": 0} for etype in ENTITY_TYPES}

    def get_id(etype: str, val: str) -> int:
        key = str(val or "").strip() or "__UNK__"
        if key not in vocabs[etype]:
            vocabs[etype][key] = len(vocabs[etype])
        return vocabs[etype][key]

    edges = defaultdict_lists()
    for j_idx, jn in enumerate(job_nodes):
        info = job_info[jn]
        for sk in (info.get("prob_skills") or {}).keys():
            s_id = get_id("skill", sk)
            edges[("job", "requires_skill", "skill")].append((j_idx, s_id))
            edges[("skill", "rev_requires_skill", "job")].append((s_id, j_idx))

        ind_id = get_id("industry", info.get("industry", "Khác"))
        edges[("job", "belongs_to_industry", "industry")].append((j_idx, ind_id))
        edges[("industry", "rev_belongs_to_industry", "job")].append((ind_id, j_idx))

        loc_id = get_id("location", info.get("city", "Unknown"))
        edges[("job", "located_in", "location")].append((j_idx, loc_id))
        edges[("location", "rev_located_in", "job")].append((loc_id, j_idx))

        role_id = get_id("role", info.get("role_can", "Other"))
        edges[("job", "has_role", "role")].append((j_idx, role_id))
        edges[("role", "rev_has_role", "job")].append((role_id, j_idx))

        exp_id = get_id("exp", info.get("exp_bucket", "Exp_Unknown"))
        edges[("job", "requires_exp", "exp")].append((j_idx, exp_id))
        edges[("exp", "rev_requires_exp", "job")].append((exp_id, j_idx))

        for q in (info.get("qualifications") or []):
            q_id = get_id("qual", q)
            edges[("job", "requires_qual", "qual")].append((j_idx, q_id))
            edges[("qual", "rev_requires_qual", "job")].append((q_id, j_idx))

    if G is not None:
        for u, v, d in G.edges(data=True):
            if d.get("rel") == "CO_OCCURS_WITH":
                su = G.nodes[u].get("label", "")
                sv = G.nodes[v].get("label", "")
                if su in vocabs["skill"] and sv in vocabs["skill"]:
                    edges[("skill", "co_occurs_with", "skill")].append(
                        (vocabs["skill"][su], vocabs["skill"][sv])
                    )

    return vocabs, edges


def defaultdict_lists():
    return {rel: [] for rel in REL_SPECS}


def _edges_to_tensors(edges: dict, device) -> dict:
    out = {}
    for rel, pairs in edges.items():
        if pairs:
            arr = np.array(pairs, dtype=np.int64).T
            out[rel] = torch.from_numpy(arr).to(device)
    return out


def train_hr_kg_gnn(
    job_nodes: list,
    job_info: dict,
    G=None,
    epochs: int = GNN_EPOCHS,
    hidden_dim: int = GNN_HIDDEN_DIM,
    text_dim: int = GNN_TEXT_DIM,
    lr: float = GNN_LR,
    checkpoint_path: str = GNN_CHECKPOINT_PATH,
    precomputed_tfidf=None,
    precomputed_X=None,
):
    """Train Heterogeneous GraphSAGE on the HR Knowledge Graph and save checkpoint."""
    if not HAS_PYG:
        raise RuntimeError("PyTorch / PyTorch Geometric is not installed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if precomputed_tfidf is not None and precomputed_X is not None:
        tfidf = precomputed_tfidf
        X_sparse = precomputed_X
    else:
        texts = [job_info[j]["text"] for j in job_nodes]
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=3,
            max_features=8000,
            sublinear_tf=True,
            lowercase=True,
        )
        X_sparse = tfidf.fit_transform(texts)
    actual_dim = min(text_dim, max(2, X_sparse.shape[1] - 1), max(2, X_sparse.shape[0] - 1))
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    X_dense = normalize(svd.fit_transform(X_sparse)).astype(np.float32)

    vocabs, edges = _extract_job_entities(job_info, job_nodes, G=G)
    entity_counts = {etype: len(v) for etype, v in vocabs.items()}
    edge_index_dict = _edges_to_tensors(edges, device)
    job_x = torch.from_numpy(X_dense).to(device)

    model = HeteroHRGNN(actual_dim, hidden_dim, entity_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_rels = [
        ("job", "requires_skill", "skill"),
        ("job", "belongs_to_industry", "industry"),
        ("job", "has_role", "role"),
        ("job", "located_in", "location"),
    ]
    active_rels = [r for r in train_rels if r in edge_index_dict and edge_index_dict[r].size(1) > 0]

    model.train()
    rng = np.random.default_rng(42)
    temperature = 0.15

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        _, z_dict = model(job_x, edge_index_dict)

        total_loss = 0.0
        for rel in active_rels:
            _, _, dst_type = rel
            eidx = edge_index_dict[rel]
            num_e = eidx.size(1)
            sample_n = min(4096, num_e)
            if num_e > sample_n:
                perm = torch.from_numpy(rng.choice(num_e, size=sample_n, replace=False)).to(device)
                pos_src = eidx[0, perm]
                pos_dst = eidx[1, perm]
            else:
                pos_src = eidx[0]
                pos_dst = eidx[1]

            neg_dst = torch.randint(0, entity_counts[dst_type], (pos_src.size(0),), device=device)

            z_s = z_dict["job"][pos_src]
            z_pos = z_dict[dst_type][pos_dst]
            z_neg = z_dict[dst_type][neg_dst]

            pos_logits = (z_s * z_pos).sum(dim=-1) / temperature
            neg_logits = (z_s * z_neg).sum(dim=-1) / temperature
            loss = -F.logsigmoid(pos_logits - neg_logits).mean()
            total_loss = total_loss + loss

        total_loss = total_loss / max(1, len(active_rels))
        total_loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(f"[GNN] Epoch {epoch:02d}/{epochs} | Loss: {total_loss.item():.4f} | Device: {device}")

    model.eval()
    with torch.no_grad():
        h1_dict, z_dict = model(job_x, edge_index_dict)

    job_embs = z_dict["job"].cpu().numpy().astype(np.float32)
    x0_entities = {
        etype: model.entity_embs[etype].weight.detach().cpu()
        for etype in ENTITY_TYPES
    }
    h1_entities = {
        etype: h1_dict[etype].detach().cpu()
        for etype in ENTITY_TYPES
    }

    checkpoint = {
        "text_dim": actual_dim,
        "hidden_dim": hidden_dim,
        "entity_counts": entity_counts,
        "vocabs": vocabs,
        "job_nodes": job_nodes,
        "job_embs": job_embs,
        "x0_entities": x0_entities,
        "h1_entities": h1_entities,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "tfidf": tfidf,
        "svd": svd,
    }
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"[GNN] Saved checkpoint to {checkpoint_path} ({len(job_nodes)} jobs, dim={hidden_dim})")
    return checkpoint


class HRGNNRuntime:
    """CPU-fast runtime wrapper for inductive CV embedding and job similarity scoring."""

    def __init__(self, checkpoint: dict):
        self.text_dim = checkpoint["text_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.entity_counts = checkpoint["entity_counts"]
        self.vocabs = checkpoint["vocabs"]
        self.job_nodes = checkpoint["job_nodes"]
        self.job_index = {jn: i for i, jn in enumerate(self.job_nodes)}
        self.job_embs = checkpoint["job_embs"]  # (N_jobs, hidden_dim) L2-normalized
        self.x0_entities = checkpoint["x0_entities"]
        self.h1_entities = checkpoint["h1_entities"]
        self.tfidf = checkpoint["tfidf"]
        self.svd = checkpoint["svd"]

        self.model = HeteroHRGNN(self.text_dim, self.hidden_dim, self.entity_counts)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @classmethod
    def load(cls, checkpoint_path: str = GNN_CHECKPOINT_PATH):
        if not HAS_PYG or not os.path.exists(checkpoint_path):
            return None
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(ckpt)

    def infer_user_embedding(
        self,
        cv_text: str,
        user_skills: dict,
        user_city: str,
        user_role: str,
        user_exp: str,
        user_industry: str = "Khác",
        user_quals: list | None = None,
    ) -> np.ndarray:
        """Compute inductive GNN embedding for an uploaded CV via 2-layer HeteroConv."""
        vec_sparse = self.tfidf.transform([norm_text(cv_text)])
        vec_dense = normalize(self.svd.transform(vec_sparse)).astype(np.float32)
        job_x = torch.from_numpy(vec_dense)

        def lookup_ids(etype: str, values: list[str]) -> list[int]:
            vocab = self.vocabs[etype]
            ids = [vocab[v] for v in values if v in vocab]
            return ids if ids else [0]

        skill_ids = lookup_ids("skill", list((user_skills or {}).keys()))
        ind_ids = lookup_ids("industry", [user_industry])
        loc_ids = lookup_ids("location", [user_city])
        role_ids = lookup_ids("role", [user_role])
        exp_ids = lookup_ids("exp", [user_exp])
        qual_ids = lookup_ids("qual", list(user_quals or []))

        def rev_edge(src_ids: list[int]) -> torch.Tensor:
            return torch.tensor([src_ids, [0] * len(src_ids)], dtype=torch.long)

        rev_edges = {
            ("skill", "rev_requires_skill", "job"): rev_edge(skill_ids),
            ("industry", "rev_belongs_to_industry", "job"): rev_edge(ind_ids),
            ("location", "rev_located_in", "job"): rev_edge(loc_ids),
            ("role", "rev_has_role", "job"): rev_edge(role_ids),
            ("exp", "rev_requires_exp", "job"): rev_edge(exp_ids),
            ("qual", "rev_requires_qual", "job"): rev_edge(qual_ids),
        }

        with torch.no_grad():
            x0_dict = {"job": self.model.job_proj(job_x), **self.x0_entities}
            out1 = self.model.conv1(x0_dict, rev_edges)
            h1_job = F.relu(self.model.norm1["job"](out1["job"] + x0_dict["job"]))

            h1_dict = {"job": h1_job, **self.h1_entities}
            out2 = self.model.conv2(h1_dict, rev_edges)
            z_job = F.normalize(self.model.norm2["job"](out2["job"] + h1_job), p=2, dim=-1)

        return z_job.squeeze(0).numpy().astype(np.float32)

    def score_jobs(self, user_emb: np.ndarray, job_nodes: list) -> dict:
        """Compute cosine similarity mapped to [0, 1] for requested job_nodes."""
        if user_emb is None:
            return {}
        sims = self.job_embs @ user_emb  # [-1, 1]
        scores = {}
        for jn in job_nodes:
            idx = self.job_index.get(jn)
            if idx is not None:
                # Map cosine similarity from [-1, 1] -> [0, 1]
                scores[jn] = float(np.clip((sims[idx] + 1.0) * 0.5, 0.0, 1.0))
        return scores

