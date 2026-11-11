# -*- coding: utf-8 -*-
"""Train the Inductive Heterogeneous HR Knowledge Graph GNN and cache serving artifacts."""

import argparse
import os
import pickle
import time

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import (
    DATASET_PATH,
    GNN_CHECKPOINT_PATH,
    GNN_EPOCHS,
    MAX_JOBS_SERVING,
)
from data_loader import load_job_dataset
from gnn_model import train_hr_kg_gnn
from graph_builder import build_job_nodes, build_skill_cooccurrence_edges
from matching import build_job_job_similar_edges

KG_CACHE_PATH = os.path.join("data", "hr_kg_cache.pkl")


def build_and_train(
    dataset_path: str = DATASET_PATH,
    max_rows: int = MAX_JOBS_SERVING,
    epochs: int = GNN_EPOCHS,
    checkpoint_path: str = GNN_CHECKPOINT_PATH,
    cache_path: str = KG_CACHE_PATH,
    with_sim_edges: bool = False,
):
    t0 = time.time()
    print(f"[1/5] Loading dataset from {dataset_path} (max_rows={max_rows})...")
    _, df = load_job_dataset(dataset_path, max_rows=max_rows if max_rows and max_rows > 0 else None)
    print(f"      Loaded {len(df)} valid job rows in {time.time() - t0:.2f}s.")

    t1 = time.time()
    print("[2/5] Building HR Knowledge Graph (NetworkX)...")
    G = nx.DiGraph()
    job_info = {}
    job_nodes, job_info = build_job_nodes(G, df, job_info, fast_mode=True)
    cooccur_cnt = build_skill_cooccurrence_edges(G, job_info)
    print(f"      Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} (CO_OCCURS_WITH: {cooccur_cnt}) in {time.time() - t1:.2f}s.")

    t2 = time.time()
    print("[3/5] Fitting TF-IDF job vectors...")
    valid_job_nodes = [j for j in job_nodes if j in job_info]
    texts = [job_info[j]["text"] for j in valid_job_nodes]
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=3,
        max_df=1.0,
        max_features=8000,
        sublinear_tf=True,
        lowercase=True,
    )
    X = normalize(tfidf.fit_transform(texts))
    IDX = {j: i for i, j in enumerate(valid_job_nodes)}

    if with_sim_edges and len(valid_job_nodes) <= 20000:
        sim_cnt = build_job_job_similar_edges(G, valid_job_nodes, job_info, IDX, X)
        print(f"      Added {sim_cnt} SIMILAR_TO edges.")
    print(f"      TF-IDF + SIMILAR_TO done in {time.time() - t2:.2f}s.")

    t3 = time.time()
    print("[4/5] Training Heterogeneous HR GraphSAGE (PyTorch Geometric)...")
    train_hr_kg_gnn(
        valid_job_nodes,
        job_info,
        G=G,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        precomputed_tfidf=tfidf,
        precomputed_X=X,
    )
    print(f"      GNN training finished in {time.time() - t3:.2f}s.")

    t4 = time.time()
    print(f"[5/5] Saving precomputed KG cache to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    cache_payload = {
        "df": df,
        "G": G,
        "job_nodes": job_nodes,
        "job_info": job_info,
        "valid_job_nodes": valid_job_nodes,
        "tfidf": tfidf,
        "X": X,
        "IDX": IDX,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Done in {time.time() - t0:.2f}s total (cache saved in {time.time() - t4:.2f}s)! System is ready to serve.")


def main():
    parser = argparse.ArgumentParser(description="Train HR Knowledge Graph GNN")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--max-rows", type=int, default=MAX_JOBS_SERVING)
    parser.add_argument("--epochs", type=int, default=GNN_EPOCHS)
    parser.add_argument("--checkpoint", default=GNN_CHECKPOINT_PATH)
    parser.add_argument("--cache", default=KG_CACHE_PATH)
    parser.add_argument("--with-sim-edges", action="store_true")
    args = parser.parse_args()

    build_and_train(
        dataset_path=args.dataset,
        max_rows=args.max_rows,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        with_sim_edges=args.with_sim_edges,
    )


if __name__ == "__main__":
    main()

