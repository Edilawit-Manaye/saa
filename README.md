================================================================================
TASK 1: RESEARCH SUMMARY — SPMINER
Representation Learning for Frequent Subgraph Mining
(Ying et al., arXiv:2402.14367, GRL+ @ ICML 2020)
================================================================================

1. SHORT SUMMARY OF THE NEURAL-SYMBOLIC PIPELINE
--------------------------------------------------------------------------------

Problem. Frequent subgraph mining seeks motifs (recurring subgraphs) that appear
most often in a target graph. Two bottlenecks make it intractable: (i) counting
occurrences of a motif is NP-hard; (ii) the number of possible k-node motifs
grows super-exponentially with k.

SPMiner pipeline. (1) Decompose: Extract node-anchored k-hop neighborhoods from
the target graph using BFS. (2) Encode: A GNN maps each neighborhood to a vector
in an order embedding space so that if graph A is a subgraph of B, then φ(A) ≤
φ(B) elementwise—i.e. A is embedded to the "lower left" of B. (3) Train once: The
GNN is trained only on synthetic graphs (Erdős–Rényi, Watts–Strogatz, Power Law
Cluster, etc.) with a max-margin order-embedding loss; it then generalizes to
any unseen graph. (4) Decoder: Motif search is a monotonic walk in embedding
space: start from a seed node, then iteratively add the node that minimizes the
total penalty m(G) over all neighborhoods. Frequency is estimated by the number
of neighborhoods whose embedding lies "top-right" of the motif, or via the soft
penalty m(G). Search strategies include greedy, beam search, and MCTS with a
neural value function.

Architecture. The encoder uses SAGE convolutions with learnable skip layers
(DenseNet-style, O(L²) weights) so that different hop-level features are
combined and oversmoothing is mitigated, improving subgraph-relation accuracy
(e.g. about 95% and over 60 AUPR on real data).


2. HOW ORDER EMBEDDINGS REPLACE EXHAUSTIVE SEARCH
--------------------------------------------------------------------------------

Exhaustive approach. Classical methods enumerate candidate k-node subgraphs
and, for each, solve subgraph isomorphism against the target. Counting is
NP-hard and the candidate set is exponentially large.

Order embedding. A partial order is defined on graphs: A ≤ B iff A is
(isomorphic to) a subgraph of B. The encoder φ is trained so that A ≤ B if and
only if φ(A) ≤ φ(B) elementwise. So "is motif G contained in neighborhood N?"
becomes an inequality check φ(G) ≤ φ(N), or E(G,N) = ||max(0, φ(G)−φ(N))||²
below a threshold—O(d) in embedding dimension d, independent of graph sizes. "How
many neighborhoods contain G?" becomes counting those N with φ(G) ≤ φ(N), or
minimizing the soft objective m(G) = sum over N of ||max(0, φ(G)−φ(N))||².

Search in embedding space. SPMiner does not enumerate all motifs. It does a
k-step monotonic walk: at each step the current motif's embedding only
increases. The next node is chosen to minimize m(G') (greedy) or via beam or
MCTS. So search is in the continuous embedding space with O(d) checks per step,
avoiding combinatorial enumeration and replacing exhaustive subgraph isomorphism
with learned geometric relations.


3. KEY TAKEAWAYS
--------------------------------------------------------------------------------

SPMiner is the first neural framework for frequent motif mining. It achieves
about 100× speedup over exact enumeration for 5–6 node motifs (e.g. 5 min vs
10 h), reliably finds 10-node motifs (beyond exact methods), and identifies
large motifs (up to about 20 nodes) with 10–100× higher frequency than
sampling baselines (MFinder, RAND-ESU). Node-anchored frequency (Definition 1)
is used for robustness and downward closure; the same pipeline also performs
well under graph-level frequency (Definition 2).

Reference: Ying, Fu, Wang, You, Wang, Leskovec. "Representation Learning for
Frequent Subgraph Mining." arXiv:2402.14367 (2024).
