#!/usr/bin/env python
"""Run the hypergraph and pairwise-projection analyses.

The construction hypergraph follows the manuscript definition:
each non-punctuation head token plus its direct non-punctuation dependents is
one hyperedge; singleton hyperedges are removed. Lexical nodes are language-
internal word types based on lower-cased lemma when available, otherwise form.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import networkit as nk

    HAVE_NETWORKIT = True
except Exception:  # pragma: no cover - depends on the local installation
    nk = None
    HAVE_NETWORKIT = False

Edge = Tuple[str, str]
Hyperedge = Tuple[str, ...]


@dataclass
class NKGraphData:
    nodes: List[str]
    idx: Dict[str, int]
    graph: object
    adj: Dict[str, Set[str]]
    degree: Dict[str, int]


@dataclass
class Token:
    idx: int
    form: str
    lemma: str
    upos: str
    head: int
    deprel: str

    @property
    def node(self) -> str:
        base = self.lemma if self.lemma and self.lemma != "_" else self.form
        return base.lower()


@dataclass
class LanguageData:
    language: str
    source_file: str
    sentences: int
    token_count: int
    nodes: Set[str]
    dep_edges: Set[Edge]
    hyperedges: List[Hyperedge]
    hyperedge_records: List[dict]
    projection_edges: Set[Edge]
    pair_source_count: Counter
    node_upos: Dict[str, str]
    node_deprel: Dict[str, str]


def norm_edge(u: str, v: str) -> Edge:
    return (u, v) if u <= v else (v, u)


def clean_deprel(label: str) -> str:
    if not label or label == "_":
        return "_"
    return label.split("@", 1)[0]


def parse_conllu(path: Path) -> Iterable[List[Token]]:
    sent: List[Token] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                    sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            if "-" in cols[0] or "." in cols[0]:
                continue
            try:
                idx = int(cols[0])
                head = int(cols[6])
            except ValueError:
                continue
            sent.append(
                Token(
                    idx=idx,
                    form=cols[1],
                    lemma=cols[2],
                    upos=cols[3],
                    head=head,
                    deprel=clean_deprel(cols[7]),
                )
            )
    if sent:
        yield sent


def build_language(path: Path) -> LanguageData:
    language = path.name.split("_", 1)[0]
    nodes: Set[str] = set()
    dep_edges: Set[Edge] = set()
    hyperedges: List[Hyperedge] = []
    hyperedge_records: List[dict] = []
    projection_edges: Set[Edge] = set()
    pair_source_count: Counter = Counter()
    upos_counts: Dict[str, Counter] = defaultdict(Counter)
    deprel_counts: Dict[str, Counter] = defaultdict(Counter)
    sentence_count = 0
    token_count = 0

    for sent in parse_conllu(path):
        sentence_count += 1
        id_to_tok = {tok.idx: tok for tok in sent}
        children: Dict[int, List[Token]] = defaultdict(list)
        for tok in sent:
            if tok.upos == "PUNCT":
                continue
            nodes.add(tok.node)
            upos_counts[tok.node][tok.upos] += 1
            deprel_counts[tok.node][tok.deprel] += 1
            token_count += 1
            if tok.head:
                children[tok.head].append(tok)

        for tok in sent:
            if tok.upos == "PUNCT" or tok.head == 0:
                continue
            head = id_to_tok.get(tok.head)
            if head is None or head.upos == "PUNCT":
                continue
            pair = norm_edge(tok.node, head.node)
            # Word-type aggregation can map a head and dependent to the same
            # lemma.  Such aggregation-induced self-loops are not edges of the
            # simple dependency graph and are also excluded by the projection.
            if pair[0] != pair[1]:
                dep_edges.add(pair)

        for head in sent:
            if head.upos == "PUNCT":
                continue
            deps = [dep for dep in children.get(head.idx, []) if dep.upos != "PUNCT"]
            members = [head.node] + [dep.node for dep in deps]
            members = sorted(set(members))
            if len(members) <= 1:
                continue
            edge = tuple(members)
            hyperedges.append(edge)
            dep_nodes = sorted({dep.node for dep in deps})
            role_combo = (
                head.upos,
                tuple(sorted(clean_deprel(dep.deprel) for dep in deps)),
            )
            hyperedge_records.append(
                {
                    "head": head.node,
                    "head_upos": head.upos,
                    "members": edge,
                    "dependents": tuple(dep_nodes),
                    "size": len(edge),
                    "role_combo": role_combo,
                    "dep_rels": tuple(sorted(clean_deprel(dep.deprel) for dep in deps)),
                    "dep_upos": tuple(sorted(dep.upos for dep in deps)),
                }
            )
            for u, v in itertools.combinations(edge, 2):
                pair = norm_edge(u, v)
                projection_edges.add(pair)
                pair_source_count[pair] += 1

    node_upos = {n: c.most_common(1)[0][0] for n, c in upos_counts.items()}
    node_deprel = {n: c.most_common(1)[0][0] for n, c in deprel_counts.items()}
    return LanguageData(
        language=language,
        source_file=str(path),
        sentences=sentence_count,
        token_count=token_count,
        nodes=nodes,
        dep_edges=dep_edges,
        hyperedges=hyperedges,
        hyperedge_records=hyperedge_records,
        projection_edges=projection_edges,
        pair_source_count=pair_source_count,
        node_upos=node_upos,
        node_deprel=node_deprel,
    )


def adjacency_from_edges(nodes: Iterable[str], edges: Iterable[Edge]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def degrees_from_edges(nodes: Iterable[str], edges: Iterable[Edge]) -> Dict[str, int]:
    adj = adjacency_from_edges(nodes, edges)
    return {n: len(ns) for n, ns in adj.items()}


def build_nk_graph(nodes: Sequence[str], edges: Iterable[Edge]) -> NKGraphData:
    if not HAVE_NETWORKIT:
        raise RuntimeError("networkit is required for this experiment runner")
    idx = {node: i for i, node in enumerate(nodes)}
    g = nk.Graph(len(nodes), weighted=False, directed=False)
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        if u != v:
            g.addEdge(idx[u], idx[v], addMissing=False)
            adj[u].add(v)
            adj[v].add(u)
    degree = {node: g.degree(i) for i, node in enumerate(nodes)}
    return NKGraphData(list(nodes), idx, g, adj, degree)


def nk_pagerank(nodes: Sequence[str], edges: Set[Edge]) -> Dict[str, float]:
    gd = build_nk_graph(nodes, edges)
    pr = nk.centrality.PageRank(gd.graph, damp=0.85, tol=1e-8)
    pr.run()
    scores = pr.scores()
    return {node: float(scores[i]) for i, node in enumerate(nodes)}


def nk_clustering_average(nodes: Sequence[str], edges: Set[Edge]) -> float:
    if len(nodes) == 0:
        return 0.0
    gd = build_nk_graph(nodes, edges)
    coeff = nk.centrality.LocalClusteringCoefficient(gd.graph)
    coeff.run()
    scores = coeff.scores()
    return float(np.mean(scores)) if scores else 0.0


def safe_corr(x: Sequence[float], y: Sequence[float], method: str) -> float:
    if len(x) < 3 or len(set(x)) <= 1 or len(set(y)) <= 1:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y, nan_policy="omit").statistic)
    return float(kendalltau(x, y, nan_policy="omit").statistic)


def top_jaccard(a: Dict[str, float], b: Dict[str, float], q: float) -> float:
    nodes = list(set(a) & set(b))
    if not nodes:
        return float("nan")
    k = max(1, int(math.ceil(len(nodes) * q)))
    ta = {n for n, _ in sorted(((n, a[n]) for n in nodes), key=lambda x: x[1], reverse=True)[:k]}
    tb = {n for n, _ in sorted(((n, b[n]) for n in nodes), key=lambda x: x[1], reverse=True)[:k]}
    return len(ta & tb) / len(ta | tb) if ta or tb else float("nan")


def neighborhood_stats(nodes: Sequence[str], dep_edges: Set[Edge], proj_edges: Set[Edge]) -> Tuple[float, float]:
    nd = defaultdict(set)
    npg = defaultdict(set)
    for u, v in dep_edges:
        if u == v:
            continue
        nd[u].add(v)
        nd[v].add(u)
    for u, v in proj_edges:
        if u == v:
            continue
        npg[u].add(v)
        npg[v].add(u)
    jvals = []
    xvals = []
    for n in nodes:
        a = nd[n]
        b = npg[n]
        union = a | b
        if union:
            jvals.append(len(a & b) / len(union))
        if a:
            xvals.append(len(b - a) / len(a))
    return float(np.mean(jvals)) if jvals else float("nan"), float(np.mean(xvals)) if xvals else float("nan")


def community_labels(nodes: Sequence[str], edges: Set[Edge]) -> List[int]:
    gd = build_nk_graph(nodes, edges)
    if gd.graph.numberOfEdges() == 0:
        return list(range(len(nodes)))
    detector = nk.community.PLM(gd.graph, refine=True)
    detector.run()
    part = detector.getPartition()
    return [int(part.subsetOf(i)) for i in range(len(nodes))]


def spectral_signature(nodes: Sequence[str], edges: Set[Edge], k: int = 30) -> np.ndarray:
    n = len(nodes)
    if n == 0:
        return np.zeros(k)
    idx = {node: i for i, node in enumerate(nodes)}
    rows = []
    cols = []
    for u, v in edges:
        if u == v:
            continue
        i, j = idx[u], idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
    if not rows:
        vals = np.zeros(min(k, n))
    else:
        a = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        deg = np.asarray(a.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        d = sparse.diags(inv_sqrt)
        lap = sparse.eye(n, format="csr") - d @ a @ d
        kk = min(k, n - 2)
        if kk <= 0:
            vals = np.linalg.eigvalsh(lap.toarray())
        else:
            vals = eigsh(lap, k=kk, which="SM", return_eigenvectors=False)
        vals = np.sort(np.asarray(vals))
    out = np.zeros(k)
    out[: min(k, len(vals))] = vals[:k]
    return out


def degree_entropy(degrees: Sequence[int]) -> float:
    counts = np.array(list(Counter(degrees).values()), dtype=float)
    return float(scipy_entropy(counts, base=2)) if len(counts) else 0.0


def degree_assortativity(edges: Set[Edge], degree: Dict[str, int]) -> float:
    if not edges:
        return float("nan")
    x = np.array([degree.get(u, 0) for u, _ in edges], dtype=float)
    y = np.array([degree.get(v, 0) for _, v in edges], dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def global_profile(nodes: Sequence[str], edges: Set[Edge]) -> Dict[str, float]:
    gd = build_nk_graph(nodes, edges)
    n = len(nodes)
    m = len(edges)
    deg = [gd.degree[node] for node in nodes]
    density = 0.0 if n < 2 else 2 * m / (n * (n - 1))
    assort = degree_assortativity(edges, gd.degree)
    cc = nk.components.ConnectedComponents(gd.graph)
    cc.run()
    comp_sizes = Counter(cc.componentOfNode(i) for i in range(n))
    largest_size = max(comp_sizes.values()) if comp_sizes else 0
    return {
        "density": density,
        "degree_entropy": degree_entropy(deg),
        "assortativity": float(assort),
        "avg_clustering": nk_clustering_average(nodes, edges),
        "largest_component_ratio": largest_size / n if n else 0.0,
        "avg_path_lcc": float("nan"),
    }


def profile_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    vals = []
    for key in sorted(a):
        x = a[key]
        y = b.get(key, float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            vals.append((x - y) ** 2)
    return float(math.sqrt(sum(vals))) if vals else float("nan")


def hyperdegree_counts(hyperedges: Sequence[Hyperedge]) -> Counter:
    c = Counter()
    for e in hyperedges:
        c.update(e)
    return c


def incidence_entropy_by_size(hyperedges: Sequence[Hyperedge]) -> Dict[str, float]:
    by_node: Dict[str, Counter] = defaultdict(Counter)
    for e in hyperedges:
        size = len(e)
        for n in e:
            by_node[n][size] += 1
    out = {}
    for n, cnt in by_node.items():
        vals = np.array(list(cnt.values()), dtype=float)
        out[n] = float(scipy_entropy(vals, base=2)) if vals.sum() else 0.0
    return out


def role_entropy(records: Sequence[dict]) -> float:
    c = Counter(str(rec["role_combo"]) for rec in records)
    vals = np.array(list(c.values()), dtype=float)
    return float(scipy_entropy(vals, base=2)) if len(vals) else 0.0


def repeated_role_rate(records: Sequence[dict]) -> float:
    c = Counter(str(rec["role_combo"]) for rec in records)
    if not records:
        return 0.0
    return sum(v for v in c.values() if v > 1) / len(records)


def hyperedge_overlap_stats(hyperedges: Sequence[Hyperedge], sample_pairs: int, rng: random.Random) -> Dict[str, float]:
    if len(hyperedges) < 2:
        return {"overlap_mean": 0.0, "overlap_p95": 0.0, "overlap_positive_rate": 0.0}
    sets = [set(e) for e in hyperedges]
    total_pairs = len(sets) * (len(sets) - 1) // 2
    if total_pairs <= sample_pairs:
        pairs = itertools.combinations(range(len(sets)), 2)
    else:
        pairs = ((rng.randrange(len(sets)), rng.randrange(len(sets))) for _ in range(sample_pairs))
    vals = []
    for i, j in pairs:
        if i == j:
            continue
        inter = len(sets[i] & sets[j])
        denom = min(len(sets[i]), len(sets[j]))
        vals.append(inter / denom if denom else 0.0)
    if not vals:
        return {"overlap_mean": 0.0, "overlap_p95": 0.0, "overlap_positive_rate": 0.0}
    return {
        "overlap_mean": float(np.mean(vals)),
        "overlap_p95": float(np.quantile(vals, 0.95)),
        "overlap_positive_rate": float(np.mean(np.array(vals) > 0)),
    }


def classify_triangles(data: LanguageData, max_triangles: int | None = None) -> Dict[str, float]:
    true_sets = [set(e) for e in set(data.hyperedges) if len(e) >= 3]
    dep = data.dep_edges
    added = data.projection_edges - data.dep_edges
    total = within = assembled = mixed = 0
    pos_coh_within = []
    pos_coh_assembled = []

    # Exact triangle enumeration without general clique enumeration.
    # General maximal-clique iterators can spend a very long time on dense
    # projected neighborhoods; ordered adjacency keeps this bounded by wedges.
    nodes = sorted(data.nodes)
    rank = {n: i for i, n in enumerate(nodes)}
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for u, v in data.projection_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in nodes:
        later = [v for v in adj[u] if rank[v] > rank[u]]
        later.sort(key=rank.get)
        later_set = set(later)
        for i, v in enumerate(later):
            later_set.discard(v)
            common = adj[v] & later_set
            for w in common:
                clique = (u, v, w)
                total += 1
                cset = set(clique)
                is_within = any(cset <= e for e in true_sets)
                edge_list = {norm_edge(a, b) for a, b in itertools.combinations(clique, 2)}
                is_mixed = bool(edge_list & dep) and bool(edge_list & added)
                upos_vals = [data.node_upos.get(n, "_") for n in clique]
                coherence = max(Counter(upos_vals).values()) / len(upos_vals)
                if is_within:
                    within += 1
                    pos_coh_within.append(coherence)
                else:
                    assembled += 1
                    pos_coh_assembled.append(coherence)
                if is_mixed:
                    mixed += 1
                if max_triangles is not None and total >= max_triangles:
                    return {
                        "triangles_total": total,
                        "triangles_sample_capped": 1,
                        "within_hyperedge_triangles": within,
                        "assembled_triangles": assembled,
                        "mixed_mechanism_triangles": mixed,
                        "within_ratio": within / total if total else float("nan"),
                        "assembled_ratio": assembled / total if total else float("nan"),
                        "mixed_ratio": mixed / total if total else float("nan"),
                        "upos_coherence_within": float(np.mean(pos_coh_within)) if pos_coh_within else float("nan"),
                        "upos_coherence_assembled": float(np.mean(pos_coh_assembled)) if pos_coh_assembled else float("nan"),
                    }
    return {
        "triangles_total": total,
        "triangles_sample_capped": 0,
        "within_hyperedge_triangles": within,
        "assembled_triangles": assembled,
        "mixed_mechanism_triangles": mixed,
        "within_ratio": within / total if total else float("nan"),
        "assembled_ratio": assembled / total if total else float("nan"),
        "mixed_ratio": mixed / total if total else float("nan"),
        "upos_coherence_within": float(np.mean(pos_coh_within)) if pos_coh_within else float("nan"),
        "upos_coherence_assembled": float(np.mean(pos_coh_assembled)) if pos_coh_assembled else float("nan"),
    }


def recoverability(data: LanguageData, max_cliques: int) -> Dict[str, float]:
    adj = adjacency_from_edges(data.nodes, data.projection_edges)
    true_unique = {tuple(sorted(e)) for e in data.hyperedges if len(e) >= 2}
    candidates: Set[Hyperedge] = set()
    for u, v in data.projection_edges:
        candidates.add(tuple(sorted((u, v))))

    # Heuristic candidate reconstruction from the projection graph. Full
    # maximal-clique enumeration is intentionally avoided: these projected
    # graphs can contain huge families of assembled cliques.
    rng = random.Random(271828 + len(data.nodes))
    target_sizes = sorted({len(e) for e in true_unique if 3 <= len(e) <= 5})
    attempts_per_size = max(30, min(max_cliques, 2000) // max(1, len(target_sizes)))
    for size in target_sizes:
        seen = 0
        for _ in range(attempts_per_size):
            cand = random_clique_of_size(adj, size, rng)
            if cand is not None:
                candidates.add(cand)
                seen += 1
            if len(candidates) >= max_cliques:
                break
        if len(candidates) >= max_cliques:
            break
    exact = candidates & true_unique
    precision = len(exact) / len(candidates) if candidates else float("nan")
    recall = len(exact) / len(true_unique) if true_unique else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")

    rng2 = random.Random(314159 + len(candidates) + len(true_unique))
    cand_list = list(candidates)
    true_list = list(true_unique)
    cand_boundary = rng2.sample(cand_list, min(1000, len(cand_list))) if cand_list else []
    true_boundary = rng2.sample(true_list, min(1000, len(true_list))) if true_list else []
    cand_sets = [set(c) for c in cand_boundary]
    true_sets_sample = [set(t) for t in true_boundary]
    boundary_errors = []
    overmerged = 0
    for cand in cand_sets:
        contained = sum(1 for ts in true_sets_sample if ts < cand)
        if contained >= 2:
            overmerged += 1
    for ts in true_sets_sample:
        best = 1.0
        for cs in cand_sets:
            union = len(ts | cs)
            if union:
                best = min(best, len(ts ^ cs) / union)
        boundary_errors.append(best)
    return {
        "sampled_candidate_cliques": len(candidates),
        "true_unique_hyperedges": len(true_unique),
        "recover_precision": precision,
        "recover_recall": recall,
        "recover_f1": f1,
        "boundary_error_mean": float(np.mean(boundary_errors)) if boundary_errors else float("nan"),
        "boundary_error_sample_n_true": len(true_sets_sample),
        "boundary_error_sample_n_candidates": len(cand_sets),
        "overmerge_rate": overmerged / len(cand_sets) if cand_sets else float("nan"),
    }


def random_clique_of_size(adj: Dict[str, Set[str]], size: int, rng: random.Random, attempts: int = 12) -> Hyperedge | None:
    nodes = list(adj)
    if size == 2:
        edges = [(u, v) for u, ns in adj.items() for v in ns if u < v]
        return tuple(sorted(rng.choice(edges))) if edges else None
    for _ in range(attempts):
        start = rng.choice(nodes)
        clique = [start]
        candidates = set(adj[start])
        while len(clique) < size and candidates:
            nxt = rng.choice(tuple(candidates))
            clique.append(nxt)
            candidates &= adj[nxt]
        if len(clique) == size:
            return tuple(sorted(clique))

        # Fallback: sample from one closed neighborhood for small sizes.
        seed = rng.choice(nodes)
        cand = [seed] + list(adj[seed])
        if len(cand) < size or size > 5:
            continue
        sample = rng.sample(cand, size)
        ok = True
        for u, v in itertools.combinations(sample, 2):
            if v not in adj.get(u, set()):
                ok = False
                break
        if ok:
            return tuple(sorted(sample))
    return None


def sampled_clique_pools(data: LanguageData, rng: random.Random, per_size: int = 100) -> Dict[int, List[Hyperedge]]:
    adj = adjacency_from_edges(data.nodes, data.projection_edges)
    true_by_size: Dict[int, List[Hyperedge]] = defaultdict(list)
    for e in data.hyperedges:
        true_by_size[len(e)].append(e)
    pools: Dict[int, List[Hyperedge]] = {}
    for size, true_edges in true_by_size.items():
        pool = set(true_edges)
        attempts = per_size * 3 if size <= 5 else per_size
        for _ in range(attempts):
            cand = random_clique_of_size(adj, size, rng)
            if cand is not None:
                pool.add(cand)
            if len(pool) >= per_size:
                break
        pools[size] = list(pool)
    return pools


def sample_null_hyperedges(data: LanguageData, rng: random.Random, pools: Dict[int, List[Hyperedge]]) -> List[Hyperedge]:
    true_sizes = [len(e) for e in data.hyperedges]
    null_edges: List[Hyperedge] = []
    for size in true_sizes:
        pool = pools.get(size, [])
        if pool:
            null_edges.append(rng.choice(pool))
    return null_edges


def pseudo_records_from_edges(edges: Sequence[Hyperedge], data: LanguageData, dep_degree: Dict[str, int]) -> List[dict]:
    recs = []
    for e in edges:
        head = max(e, key=lambda n: dep_degree.get(n, 0))
        deps = [n for n in e if n != head]
        recs.append(
            {
                "head": head,
                "head_upos": data.node_upos.get(head, "_"),
                "members": e,
                "dependents": tuple(deps),
                "size": len(e),
                "role_combo": (
                    data.node_upos.get(head, "_"),
                    tuple(sorted(data.node_deprel.get(n, "_") for n in deps)),
                ),
                "dep_rels": tuple(sorted(data.node_deprel.get(n, "_") for n in deps)),
                "dep_upos": tuple(sorted(data.node_upos.get(n, "_") for n in deps)),
            }
        )
    return recs


def hypergraph_observables(
    hyperedges: Sequence[Hyperedge],
    records: Sequence[dict],
    rng: random.Random,
    sample_pairs: int,
) -> Dict[str, float]:
    sizes = [len(e) for e in hyperedges]
    hd = hyperdegree_counts(hyperedges)
    inc_ent = incidence_entropy_by_size(hyperedges)
    out = {
        "hyperedge_count": len(hyperedges),
        "unique_hyperedge_count": len(set(hyperedges)),
        "mean_hyperedge_size": float(np.mean(sizes)) if sizes else 0.0,
        "max_hyperedge_size": max(sizes) if sizes else 0,
        "hyperdegree_mean": float(np.mean(list(hd.values()))) if hd else 0.0,
        "hyperdegree_entropy": degree_entropy(list(hd.values())),
        "incidence_entropy_mean": float(np.mean(list(inc_ent.values()))) if inc_ent else 0.0,
        "role_entropy": role_entropy(records),
        "repeated_role_rate": repeated_role_rate(records),
    }
    out.update(hyperedge_overlap_stats(hyperedges, sample_pairs, rng))
    return out


def null_experiment(data: LanguageData, n_null: int, sample_pairs: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    dep_degree = degrees_from_edges(data.nodes, data.dep_edges)
    empirical = hypergraph_observables(data.hyperedges, data.hyperedge_records, rng, sample_pairs)
    empirical["language"] = data.language
    empirical["kind"] = "empirical"
    rows = []
    pools = sampled_clique_pools(data, rng)
    for i in range(n_null):
        null_edges = sample_null_hyperedges(data, rng, pools)
        null_records = pseudo_records_from_edges(null_edges, data, dep_degree)
        obs = hypergraph_observables(null_edges, null_records, rng, sample_pairs)
        obs["language"] = data.language
        obs["kind"] = "null"
        obs["null_id"] = i
        rows.append(obs)
    null_df = pd.DataFrame(rows)
    zrows = []
    for metric, emp_val in empirical.items():
        if metric in {"language", "kind"} or not isinstance(emp_val, (int, float, np.number)):
            continue
        vals = null_df[metric].dropna().astype(float).values if metric in null_df else np.array([])
        if len(vals) == 0:
            continue
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        mean = float(np.mean(vals))
        z = (float(emp_val) - mean) / sd if sd and math.isfinite(sd) else float("nan")
        p_hi = (1 + np.sum(vals >= float(emp_val))) / (1 + len(vals))
        p_lo = (1 + np.sum(vals <= float(emp_val))) / (1 + len(vals))
        p_two = min(1.0, 2 * min(p_hi, p_lo))
        zrows.append(
            {
                "language": data.language,
                "metric": metric,
                "empirical": float(emp_val),
                "null_mean": mean,
                "null_sd": sd,
                "z": z,
                "p_two_sided": p_two,
            }
        )
    return pd.DataFrame([empirical]), pd.DataFrame(zrows), null_df


def feature_entropy(labels: List[str]) -> float:
    c = Counter(labels)
    vals = np.array(list(c.values()), dtype=float)
    return float(scipy_entropy(vals, base=2)) if len(vals) else 0.0


def candidate_features(edge: Hyperedge, data: LanguageData, dep_deg: Dict[str, int], proj_deg: Dict[str, int], hd: Counter) -> Dict[str, float]:
    pairs = [norm_edge(u, v) for u, v in itertools.combinations(edge, 2)]
    weights = [data.pair_source_count.get(p, 0) for p in pairs]
    pdeg = np.array([proj_deg.get(n, 0) for n in edge], dtype=float)
    ddeg = np.array([dep_deg.get(n, 0) for n in edge], dtype=float)
    hdeg = np.array([hd.get(n, 0) for n in edge], dtype=float)
    upos = [data.node_upos.get(n, "_") for n in edge]
    deprel = [data.node_deprel.get(n, "_") for n in edge]
    return {
        "size": len(edge),
        "pair_weight_mean": float(np.mean(weights)) if weights else 0.0,
        "pair_weight_max": float(np.max(weights)) if weights else 0.0,
        "proj_degree_mean": float(pdeg.mean()) if len(pdeg) else 0.0,
        "proj_degree_std": float(pdeg.std()) if len(pdeg) else 0.0,
        "dep_degree_mean": float(ddeg.mean()) if len(ddeg) else 0.0,
        "dep_degree_std": float(ddeg.std()) if len(ddeg) else 0.0,
        "upos_entropy": feature_entropy(upos),
        "deprel_entropy": feature_entropy(deprel),
        "upos_unique": len(set(upos)),
        "deprel_unique": len(set(deprel)),
        "hyperdegree_mean": float(hdeg.mean()) if len(hdeg) else 0.0,
        "hyperdegree_std": float(hdeg.std()) if len(hdeg) else 0.0,
        "hyperdegree_min": float(hdeg.min()) if len(hdeg) else 0.0,
    }


def prediction_dataset(data: LanguageData, seed: int, max_neg_per_size: int) -> pd.DataFrame:
    rng = random.Random(seed)
    true_edges = {tuple(sorted(e)) for e in data.hyperedges if len(e) >= 3}
    adj = adjacency_from_edges(data.nodes, data.projection_edges)
    dep_deg = degrees_from_edges(data.nodes, data.dep_edges)
    proj_deg = degrees_from_edges(data.nodes, data.projection_edges)
    hd = hyperdegree_counts(data.hyperedges)
    rows = []
    for e in true_edges:
        row = candidate_features(e, data, dep_deg, proj_deg, hd)
        row["language"] = data.language
        row["label"] = 1
        rows.append(row)
    sizes = Counter(len(e) for e in true_edges if len(e) <= 5)
    for size, pos_count in sizes.items():
        target = min(max_neg_per_size, max(pos_count, 1))
        seen = set()
        attempts = 0
        while len(seen) < target and attempts < target * 30:
            attempts += 1
            cand = random_clique_of_size(adj, size, rng)
            if cand is None or cand in true_edges or cand in seen:
                continue
            seen.add(cand)
        for e in seen:
            row = candidate_features(e, data, dep_deg, proj_deg, hd)
            row["language"] = data.language
            row["label"] = 0
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_prediction(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df["label"].nunique() < 2 or len(df) < 20:
        return pd.DataFrame()
    feature_groups = {
        "P-only": [
            "size",
            "pair_weight_mean",
            "pair_weight_max",
            "proj_degree_mean",
            "proj_degree_std",
            "dep_degree_mean",
            "dep_degree_std",
        ],
        "P+Role": [
            "size",
            "pair_weight_mean",
            "pair_weight_max",
            "proj_degree_mean",
            "proj_degree_std",
            "dep_degree_mean",
            "dep_degree_std",
            "upos_entropy",
            "deprel_entropy",
            "upos_unique",
            "deprel_unique",
        ],
        "P+Incidence": [
            "size",
            "pair_weight_mean",
            "pair_weight_max",
            "proj_degree_mean",
            "proj_degree_std",
            "dep_degree_mean",
            "dep_degree_std",
            "hyperdegree_mean",
            "hyperdegree_std",
            "hyperdegree_min",
        ],
        "Full": [
            "size",
            "pair_weight_mean",
            "pair_weight_max",
            "proj_degree_mean",
            "proj_degree_std",
            "dep_degree_mean",
            "dep_degree_std",
            "upos_entropy",
            "deprel_entropy",
            "upos_unique",
            "deprel_unique",
            "hyperdegree_mean",
            "hyperdegree_std",
            "hyperdegree_min",
        ],
    }
    y = df["label"].astype(int).values
    min_class = int(np.min(np.bincount(y)))
    splits = min(5, min_class)
    if splits < 2:
        return pd.DataFrame()
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    rows = []
    for name, cols in feature_groups.items():
        x = df[cols].fillna(0.0).astype(float).values
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17),
        )
        prob = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "n": len(df),
                "positive": int(y.sum()),
                "negative": int((1 - y).sum()),
                "auc": roc_auc_score(y, prob),
                "f1": f1_score(y, pred),
            }
        )
    # A tree model is useful as a non-linear robustness check for the full feature set.
    cols = feature_groups["Full"]
    x = df[cols].fillna(0.0).astype(float).values
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=17,
        n_jobs=-1,
    )
    prob = cross_val_predict(rf, x, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    rows.append(
        {
            "model": "Full-RF",
            "n": len(df),
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
            "auc": roc_auc_score(y, prob),
            "f1": f1_score(y, pred),
        }
    )
    return pd.DataFrame(rows)


def layer_similarity(data: LanguageData) -> Dict[str, float]:
    nodes = sorted(data.nodes)
    dep_deg = degrees_from_edges(nodes, data.dep_edges)
    proj_deg = degrees_from_edges(nodes, data.projection_edges)
    hd = hyperdegree_counts(data.hyperedges)
    dep_pr = nk_pagerank(nodes, data.dep_edges)
    proj_pr = nk_pagerank(nodes, data.projection_edges)
    deg_d = [dep_deg.get(n, 0) for n in nodes]
    deg_p = [proj_deg.get(n, 0) for n in nodes]
    deg_h = [hd.get(n, 0) for n in nodes]
    pr_d = [dep_pr.get(n, 0.0) for n in nodes]
    pr_p = [proj_pr.get(n, 0.0) for n in nodes]
    neigh_j, neigh_x = neighborhood_stats(nodes, data.dep_edges, data.projection_edges)
    labels_d = community_labels(nodes, data.dep_edges)
    labels_p = community_labels(nodes, data.projection_edges)
    prof_d = global_profile(nodes, data.dep_edges)
    prof_p = global_profile(nodes, data.projection_edges)
    spec_d = spectral_signature(nodes, data.dep_edges)
    spec_p = spectral_signature(nodes, data.projection_edges)
    dep_edges = data.dep_edges
    proj_edges = data.projection_edges
    inter = len(dep_edges & proj_edges)
    union = len(dep_edges | proj_edges)
    out = {
        "language": data.language,
        "source_file": data.source_file,
        "sentences": data.sentences,
        "tokens_nonpunct": data.token_count,
        "nodes": len(nodes),
        "dep_edges": len(dep_edges),
        "projection_edges": len(proj_edges),
        "added_projection_edges": len(proj_edges - dep_edges),
        "hyperedge_occurrences": len(data.hyperedges),
        "unique_hyperedges": len(set(data.hyperedges)),
        "projection_edge_jaccard": inter / union if union else float("nan"),
        "added_edge_ratio": len(proj_edges - dep_edges) / len(dep_edges) if dep_edges else float("nan"),
        "dep_proj_degree_spearman": safe_corr(deg_d, deg_p, "spearman"),
        "dep_hyperdegree_spearman": safe_corr(deg_d, deg_h, "spearman"),
        "pagerank_spearman": safe_corr(pr_d, pr_p, "spearman"),
        "dep_proj_degree_kendall": safe_corr(deg_d, deg_p, "kendall"),
        "dep_hyperdegree_kendall": safe_corr(deg_d, deg_h, "kendall"),
        "top5_degree_jaccard": top_jaccard(dep_deg, proj_deg, 0.05),
        "top10_degree_jaccard": top_jaccard(dep_deg, proj_deg, 0.10),
        "top20_degree_jaccard": top_jaccard(dep_deg, proj_deg, 0.20),
        "top10_pagerank_jaccard": top_jaccard(dep_pr, proj_pr, 0.10),
        "neighborhood_jaccard_mean": neigh_j,
        "neighborhood_expansion_mean": neigh_x,
        "community_nmi": normalized_mutual_info_score(labels_d, labels_p),
        "community_ari": adjusted_rand_score(labels_d, labels_p),
        "spectral_l2_distance": float(np.linalg.norm(spec_d - spec_p)),
        "global_profile_distance": profile_distance(prof_d, prof_p),
    }
    for prefix, prof in [("dep", prof_d), ("proj", prof_p)]:
        for key, val in prof.items():
            out[f"{prefix}_{key}"] = val
    return out


def write_manifest(out_dir: Path, args: argparse.Namespace, files: List[Path]) -> None:
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "executable": sys.executable,
        "networkit_available": HAVE_NETWORKIT,
        "networkit_version": getattr(nk, "__version__", None) if HAVE_NETWORKIT else None,
        "arguments": arguments,
        "outputs": [str(p) for p in files],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(out_dir: Path, tables: Dict[str, pd.DataFrame], started: float) -> None:
    sim = tables["layer_similarity"]
    tri = tables["clique_taxonomy"]
    rec = tables["recoverability"]
    nullz = tables["null_zscores"]
    pred = tables["prediction_scores"]
    lines = []
    lines.append("# Hypergraph projection analysis summary")
    lines.append("")
    lines.append(f"- Languages processed: {len(sim)}")
    lines.append("- Network backend: networkit")
    lines.append(f"- Runtime seconds: {time.time() - started:.1f}")
    lines.append("")
    lines.append("## Cross-layer similarity")
    for col in [
        "projection_edge_jaccard",
        "dep_proj_degree_spearman",
        "pagerank_spearman",
        "neighborhood_jaccard_mean",
        "community_nmi",
        "spectral_l2_distance",
    ]:
        vals = sim[col].dropna().astype(float)
        if len(vals):
            lines.append(f"- {col}: mean={vals.mean():.4f}, median={vals.median():.4f}")
    lines.append("")
    lines.append("## Clique-hyperedge non-equivalence")
    for col in ["within_ratio", "assembled_ratio", "mixed_ratio"]:
        vals = tri[col].dropna().astype(float)
        if len(vals):
            lines.append(f"- {col}: mean={vals.mean():.4f}, median={vals.median():.4f}")
    for col in ["recover_precision", "recover_recall", "recover_f1", "boundary_error_mean"]:
        vals = rec[col].dropna().astype(float)
        if len(vals):
            lines.append(f"- {col}: mean={vals.mean():.4f}, median={vals.median():.4f}")
    lines.append("")
    lines.append("## Same-projection null")
    if not nullz.empty:
        show = (
            nullz.groupby("metric")["z"]
            .agg(["mean", "median", "count"])
            .reset_index()
            .sort_values("mean", key=lambda s: s.abs(), ascending=False)
        )
        for _, row in show.head(10).iterrows():
            lines.append(
                f"- {row['metric']}: z_mean={row['mean']:.3f}, z_median={row['median']:.3f}, n={int(row['count'])}"
            )
    lines.append("")
    lines.append("## Boundary prediction")
    if not pred.empty:
        for _, row in pred.iterrows():
            lines.append(f"- {row['model']}: AUC={row['auc']:.4f}, F1={row['f1']:.4f}, n={int(row['n'])}")
    lines.append("")
    lines.append("See CSV files in this run directory for per-language values and null distributions.")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dependency hypergraphs with their pairwise projections."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="directory containing one or more CoNLL-U files",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="parent directory for analysis runs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="output subdirectory name (default: timestamped name)",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=100,
        help="projection-preserving null samples per treebank (default: 100)",
    )
    parser.add_argument("--sample-overlap-pairs", type=int, default=20_000)
    parser.add_argument("--max-triangles", type=int, default=200_000)
    parser.add_argument("--max-recover-cliques", type=int, default=200_000)
    parser.add_argument("--max-neg-per-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_505)
    args = parser.parse_args()

    if not HAVE_NETWORKIT:
        parser.error("networkit is required; install the packages in requirements.txt")

    started = time.time()
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "logs").mkdir()
    (out_dir / "tables").mkdir()
    progress_log = out_dir / "logs" / "progress.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with progress_log.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    files = sorted(args.data_dir.glob("*.conllu"))
    if not files:
        raise FileNotFoundError(f"No .conllu files found in {args.data_dir}")

    layer_rows = []
    triangle_rows = []
    recover_rows = []
    emp_rows = []
    null_z_rows = []
    null_dist_rows = []
    pred_rows = []
    pred_data_rows = []

    for i, path in enumerate(files, 1):
        lang_started = time.time()
        log(f"[{i}/{len(files)}] {path.name}")
        data = build_language(path)
        log(f"  built language in {time.time() - lang_started:.1f}s")
        t = time.time()
        layer_rows.append(layer_similarity(data))
        log(f"  layer similarity in {time.time() - t:.1f}s")
        t = time.time()
        tri = classify_triangles(data, max_triangles=args.max_triangles)
        tri["language"] = data.language
        triangle_rows.append(tri)
        log(f"  triangle taxonomy in {time.time() - t:.1f}s")
        t = time.time()
        rec = recoverability(data, args.max_recover_cliques)
        rec["language"] = data.language
        recover_rows.append(rec)
        log(f"  recoverability in {time.time() - t:.1f}s")
        t = time.time()
        emp, nullz, nulld = null_experiment(
            data,
            n_null=args.n_null,
            sample_pairs=args.sample_overlap_pairs,
            seed=args.seed + i,
        )
        emp_rows.append(emp)
        null_z_rows.append(nullz)
        if not nulld.empty:
            null_dist_rows.append(nulld)
        log(f"  same-projection null in {time.time() - t:.1f}s")
        t = time.time()
        pdf = prediction_dataset(data, seed=args.seed + 1000 + i, max_neg_per_size=args.max_neg_per_size)
        if not pdf.empty:
            pred_data_rows.append(pdf)
        log(f"  prediction candidates in {time.time() - t:.1f}s")
        log(f"  language done in {time.time() - lang_started:.1f}s")

    tables = {
        "layer_similarity": pd.DataFrame(layer_rows),
        "clique_taxonomy": pd.DataFrame(triangle_rows),
        "recoverability": pd.DataFrame(recover_rows),
        "empirical_hypergraph_metrics": pd.concat(emp_rows, ignore_index=True) if emp_rows else pd.DataFrame(),
        "null_zscores": pd.concat(null_z_rows, ignore_index=True) if null_z_rows else pd.DataFrame(),
        "null_distributions": pd.concat(null_dist_rows, ignore_index=True) if null_dist_rows else pd.DataFrame(),
    }
    pred_data = pd.concat(pred_data_rows, ignore_index=True) if pred_data_rows else pd.DataFrame()
    tables["prediction_dataset"] = pred_data
    tables["prediction_scores"] = evaluate_prediction(pred_data)

    output_files = []
    for name, df in tables.items():
        path = out_dir / "tables" / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        output_files.append(path)

    summarize(out_dir, tables, started)
    output_files.append(out_dir / "summary.md")
    write_manifest(out_dir, args, output_files)
    output_files.append(out_dir / "manifest.json")
    latest = args.out_root / "latest_run.txt"
    latest.write_text(str(out_dir), encoding="utf-8")
    print(f"Done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
