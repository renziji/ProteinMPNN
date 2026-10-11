# ProteinMPNN — Code Wiki

Comprehensive documentation of the ProteinMPNN repository: architecture, module responsibilities, key classes/functions, dependency relationships, and run instructions.

> **Reference**: Dauparas et al., *"Robust deep learning–based protein sequence design using ProteinMPNN"*, Science 378 (6615), 2022.
> Part of the graph/message-passing code is adopted from [jingraham/neurips19-graph-protein-design](https://github.com/jingraham/neurips19-graph-protein-design).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [System Architecture](#3-system-architecture)
4. [Core Library — `protein_mpnn_utils.py`](#4-core-library--protein_mpnn_utilspy)
5. [Inference Entry Point — `protein_mpnn_run.py`](#5-inference-entry-point--protein_mpnn_runpy)
6. [Training Pipeline — `training/`](#6-training-pipeline--training)
7. [Helper Scripts — `helper_scripts/`](#7-helper-scripts--helper_scripts)
8. [Dependency Relationships](#8-dependency-relationships)
9. [Model Weights](#9-model-weights)
10. [Running the Project](#10-running-the-project)
11. [Output Format Reference](#11-output-format-reference)
12. [Key Domain Concepts](#12-key-domain-concepts)

---

## 1. Project Overview

**ProteinMPNN** is a deep-learning tool for **protein sequence design**: given a fixed protein 3D backbone (N, CA, C, O atom coordinates, or CA-only), it generates amino-acid sequences predicted to fold into that backbone.

It is a **graph-conditioned autoregressive language model**:

- The backbone is converted into a *k*-nearest-neighbor graph (k = 48 in released models) where nodes = residues and edges encode inter-residue geometry (distances via radial basis functions + relative positional encodings).
- A **message-passing encoder** builds structure-aware residue representations.
- An **autoregressive decoder** samples amino acids one residue at a time in a **randomized decoding order** (not left-to-right), conditioned on both the backbone and previously sampled residues.
- Sequence design is cast as learning `P(sequence | backbone)`; sampling at different temperatures trades diversity vs. reliability.

Typical use: after designing a backbone with e.g. RFdiffusion/hallucination, run ProteinMPNN to obtain 1–100s of candidate sequences, then validate with structure prediction (AlphaFold2/RoseTTAFold).

---

## 2. Repository Layout

```
ProteinMPNN/
├── protein_mpnn_run.py          # Inference entry point (CLI)
├── protein_mpnn_utils.py        # Core library: parsing, featurization, model, losses
├── vanilla_model_weights/       # Pre-trained full-backbone models (v_48_002/010/020/030.pt)
├── soluble_model_weights/       # Models trained on soluble proteins only (v_48_002/010/020/030.pt)
├── ca_model_weights/            # CA-only models (v_48_002/010/020.pt) + excluded_PDBs.csv
├── helper_scripts/              # CLI tools to build constraint dictionaries (JSONL)
│   ├── parse_multiple_chains.py     # Folder of PDBs -> parsed.jsonl  (ALWAYS first step)
│   ├── assign_fixed_chains.py       # Which chains to design vs. fix
│   ├── make_fixed_positions_dict.py # Fixed / non-fixed residue positions
│   ├── make_tied_positions_dict.py  # Tied positions (e.g. homooligomer symmetry)
│   ├── make_pos_neg_tied_positions_dict.py  # Tied positions w/ positive+negative design
│   ├── make_bias_AA.py              # Global amino-acid composition bias
│   ├── make_bias_per_res_dict.py    # Per-residue bias
│   ├── make_pssm_input_dict.py      # PSSM bias input from npz profiles
│   └── other_tools/
│       ├── make_pssm_dict.py        # PSSM utilities
│       └── make_omit_AA.py          # Per-position amino-acid omission
├── examples/                    # SLURM shell scripts demonstrating all workflows
├── inputs/                      # Example inputs (PDB_monomers/, PDB_complexes/,
│   │                            #  PDB_homooligomers/, PSSM_inputs/)
├── outputs/                     # Example outputs (seqs/*.fa, scores, probs ...)
├── colab_notebooks/             # Google Colab demos (quickdemo, ca_only_quickdemo, wAF2)
├── training/                    # Full retraining pipeline
│   ├── training.py              # Training loop entry point (CLI)
│   ├── model_utils.py           # Training featurize(), losses, model, NoamOpt
│   ├── utils.py                 # Dataset/loaders for PDB training data
│   ├── parse_cif_noX.py         # Training-data preparation from mmCIF (TM-align etc.)
│   ├── submit_exp_020.sh        # SLURM submission example
│   ├── exp_020/                 # Sample training outputs (log.txt, model_weights/)
│   └── README.md                # Training data format documentation
├── README.md
├── LICENSE
└── .gitignore
```

---

## 3. System Architecture

### 3.1 High-level data flow (inference)

```
 PDB file(s)                          constraint JSONL dicts
     │                               (chains / fixed positions /
     ▼                                bias / PSSM / tied positions)
 parse_PDB() ──► StructureDatasetPDB ──────┐
      or                                   ▼
 pre-parsed .jsonl ──► StructureDataset ──► tied_featurize()  ──► torch tensors
                                                                  X (coords), S (native seq),
                                                                  mask, chain_M, chain_M_pos,
                                                                  chain_encoding, residue_idx,
                                                                  omit_AA_mask, pssm_*, tied_beta
                                                                       │
                                                                       ▼
                                              ┌──────────────── ProteinMPNN ────────────────┐
                                              │ ProteinFeatures: kNN graph + RBF + pos.enc. │
                                              │ EncLayer ×3  (structure self-attention)     │
                                              │ DecLayer ×3  (autoregressive, random order) │
                                              └─────────────────────────────────────────────┘
                                                       │                        │
                                     forward()/sample()/tied_sample()    conditional_probs()
                                                       │                        unconditional_probs()
                                                       ▼
                                     FASTA (seqs/*.fa) + optional npz (scores/probs)
```

### 3.2 Model internals (one forward pass)

```
X [B,L,4,3] ─► ProteinFeatures ─► E [B,L,48,edge_in], E_idx [B,L,48]
                                      │ W_e
h_V = 0 ──────────────► h_E [B,L,48,128]
        ┌──────────────────────────┐
        │ EncLayer ×3              │  message passing over kNN graph
        └──────────────────────────┘
h_S = W_s(S)  ──► sequence embeddings
        ┌──────────────────────────┐
        │ DecLayer ×3              │  autoregressive: masked attention based on a
        └──────────────────────────┘  random decoding order (chain_M-weighted)
                    │ W_out + log_softmax
                    ▼
        log_probs [B,L,21]        (alphabet: ACDEFGHIKLMNPQRSTVWYX)
```

Key architectural parameters (default inference config): `hidden_dim = 128`, `num_encoder_layers = num_decoder_layers = 3`, `num_letters = 21`, `k_neighbors = 48` (read from checkpoint `num_edges`), `augment_eps = args.backbone_noise`.

---

## 4. Core Library — `protein_mpnn_utils.py`

Single-module library (~1380 lines) imported by the inference script. Organized into six functional groups.

### 4.1 Structure/sequence parsing

| Function | Location | Description |
|---|---|---|
| `parse_fasta(filename, limit=-1, omit=[])` | L18 | Read a FASTA file → `(headers[], sequences[])`. `omit` filters characters (used to strip `/` chain separators). |
| `parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None)` | L54 | Low-level fixed-column PDB line parser. Converts `HETATM MSE` (selenomethionine) to `MET`. Handles insertion codes. Returns `(xyz [L,n_atoms,3], sequence)`; missing atoms are `NaN`-filled, missing residues map to gap char `-` (index 20). |
| `parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False)` | L139 | Iterate over chain letters (`A–Z, a–z` + numeric fallback, up to 556 chains) and build the **canonical parsed dict** consumed everywhere else: |

Canonical parsed-PDB dict format (one per PDB):

```python
{
  'name': '<filename without .pdb>',
  'num_of_chains': int,
  'seq': '<concatenated chain sequences>',
  'seq_chain_<X>': '<sequence of chain X>',
  'coords_chain_<X>': {
      'N_chain_<X>' : [[x,y,z], ...],   # [L,3] lists
      'CA_chain_<X>': [[x,y,z], ...],
      'C_chain_<X>' : [[x,y,z], ...],
      'O_chain_<X>' : [[x,y,z], ...],
      # or, with ca_only=True:  'CA_chain_<X>' only
  },
}
```

### 4.2 Featurization

| Function | Location | Description |
|---|---|---|
| `tied_featurize(batch, device, chain_dict, fixed_position_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False)` | L191 | **The batch packer** for inference. Pads a variable-length batch of parsed dicts into aligned torch tensors and applies *every* design constraint. Returns a 20-tuple: `X, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta`. |

Semantics of produced tensors:

| Tensor | Shape | Meaning |
|---|---|---|
| `X` | `[B, L, 4, 3]` (or `[B,L,1,3]` CA-only) | Backbone coordinates; missing/padded → 0 after `isnan` masking |
| `S` | `[B, L]` | Native (ground-truth) sequence as integer labels |
| `mask` | `[B, L]` | 1.0 where real atoms exist (finite coords) |
| `chain_M` | `[B, L]` | **Design mask**: 1.0 = residue to be redesigned (masked chain), 0.0 = fixed/visible chain |
| `chain_M_pos` | `[B, L]` | Position-level design mask after applying `fixed_positions_dict` (non-fixed positions → 0) |
| `chain_encoding_all` | `[B, L]` | Integer chain id, 1,2,3,... in order masked-then-visible |
| `residue_idx` | `[B, L]` | Position index that *jumps between chains*: `100*(chain_id-1) + i` (prevents positional encodings from chaining across chains) |
| `omit_AA_mask` | `[B, L, 21]` | 1 = forbid this amino acid at this position |
| `dihedral_mask` | `[B, L, 3]` | φ/ψ/ω feasibility mask (adjacent-residue jumps) |
| `pssm_coef/bias/log_odds` | `[B,L]` / `[B,L,21]` / `[B,L,21]` | PSSM blending controls (log-odds initialized to 10000 = "allowed") |
| `tied_beta` | `[L]` | Weight per position for tied (symmetry) sampling logits |
| `*list_list` | python lists | Chain bookkeeping used to reconstruct per-chain FASTA output |

### 4.3 Losses & scoring

| Function | Location | Description |
|---|---|---|
| `_scores(S, log_probs, mask)` | L39 | Per-protein mean negative log-likelihood of sequences under the model (masked). Used both for `score` (designed residues only, `mask*chain_M*chain_M_pos`) and `global_score` (all residues, `mask`). |
| `_S_to_seq(S, mask)` | L49 | Integer labels → amino-acid string (skips masked-out positions). |
| `loss_nll(S, log_probs, mask)` | L440 | Plain NLL loss + masked average (validation metric). |
| `loss_smoothed(S, log_probs, mask, weight=0.1)` | L450 | Training loss with label smoothing: one-hot target + `0.1/21` added, renormalized. |

### 4.4 Dataset containers

| Class | Location | Description |
|---|---|---|
| `StructureDataset` | L462 | Reads a `.jsonl` file (one parsed-PDB dict per line). Filters entries containing characters outside the alphabet (`ACDEFGHIKLMNPQRSTVWYX-`) or longer than `max_length`. Supports `truncate` for quick tests. Iterable. |
| `StructureDatasetPDB` | L517 | Same filtering logic but wraps an in-memory list from `parse_PDB` (single-PDB CLI path). |
| `StructureLoader` | L559 | Custom batch iterator: sorts proteins by length, groups them into batches whose `length × count ≤ batch_size` (token-bucket batching), then shuffles batch order each epoch. Yields lists of raw dicts (padding happens later in `tied_featurize`). |

### 4.5 Graph gather operations (sparse-graph plumbing)

| Function | Location | Description |
|---|---|---|
| `gather_edges(edges, neighbor_idx)` | L595 | `[B,N,N,C]` + `[B,N,K]` → neighbor edge features `[B,N,K,C]` |
| `gather_nodes(nodes, neighbor_idx)` | L601 | `[B,N,C]` + `[B,N,K]` → `[B,N,K,C]` |
| `gather_nodes_t(nodes, neighbor_idx)` | L611 | `[B,N,C]` + `[B,K]` → `[B,K,C]` |
| `cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)` | L617 | Concatenate node features onto gathered neighbor features along the last axis (used to build `h_ES`, `h_EXV` decoder inputs) |

`E_idx` is the kNN index tensor `[B, L, K]` — everything in the model is sparse over these K=48 neighbors rather than the full L×L matrix.

### 4.6 Neural-network components

| Class | Location | Description |
|---|---|---|
| `PositionWiseFeedForward` | L713 | `Linear(128→512)` → GELU → `Linear(512→128)` |
| `PositionalEncodings` | L724 | Relative sequence-offset embedding: `clip(offset + 32, 0..64)` one-hot (65 classes) → linear to 16 dims. Offset is *masked across chains* (only same-chain pairs get real offsets). |
| `EncLayer` | L623 | One encoder block: (1) edge-message aggregation over neighbors `W1→W2→W3` (GELU MLP), scaled by 1/30, residual + LayerNorm; (2) position-wise FFN; (3) second message MLP updates **edge** features `h_E` too. Input `h_E` has `num_in = hidden*2` (edge + node cat). |
| `DecLayer` | L671 | One decoder block, same message-passing scheme but updates **node** features only (`num_in = hidden*3`: pre-attached node+edge+sequence embeddings); autoregressive masking is applied by the caller through `mask_attend`. |
| `ProteinFeatures` | L920 | **Full-backbone featurizer.** Computes virtual Cβ from N/CA/C (`Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA`), builds kNN graph on CA distances (top-48), then 25 atom-pair distance sets (N,CA,C,O,Cb × ordered pairs) → 16 RBFs each (μ∈[2,22]Å) = 400 dims + 16 positional dims → linear to `edge_features`. Node input dim 6 declared but node embedding unused (`h_V` starts at zeros). |
| `CA_ProteinFeatures` | L739 | **CA-only featurizer** (`--ca_only`). Edge features = positional (16) + 9 CA-CA offset RBF sets (144) + local-frame orientations (7 = direction 3 + quaternion 4). Includes backbone-orientation machinery `_orientations_coarse` and rotation→quaternion conversion `_quaternions`. |
| `ProteinMPNN` | L1019 | **The model.** See below. |

### 4.7 `ProteinMPNN` class — key methods

```python
ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
            hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
            vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False)
```

| Method | Description |
|---|---|
| `forward(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False, decoding_order=None)` | Teacher-forced log-likelihood `log P(S | X)` used for scoring and for re-scoring sampled sequences. Decoding order = `argsort((chain_M + 1e-4) * |randn|)` — designed positions get a random order; fixed/visible residues are effectively decoded *last*. The backward-mask (`mask_bw`) hides not-yet-decoded residues from the decoder; forward-mask (`mask_fw`) streams fixed backbone context. |
| `sample(X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask, temperature, omit_AAs_np, bias_AAs_np, chain_M_pos, omit_AA_mask, pssm_coef, pssm_bias, pssm_multi, pssm_log_odds_flag, pssm_log_odds_mask, pssm_bias_flag, bias_by_res)` | **Autoregressive sequence sampling** (one residue per step over `N_nodes`): builds decoder state incrementally with `scatter_`, computes softmax logits scaled by `1/temperature`, applies in order: global AA omission (−1e8), global AA bias, per-residue bias (all `/temperature`), PSSM probability mixing (`(1−c)·probs + c·pssm`), PSSM log-odds masking, per-position AA omission; then `torch.multinomial` draw. Fixed residues (`chain_mask=0`) copy `S_true`. Returns `{"S", "probs", "decoding_order"}`. |
| `tied_sample(..., tied_pos, tied_beta, ...)` | Symmetry-constrained variant: groups tied positions into blocks (per `tied_positions_jsonl`); all positions in a block are decoded **simultaneously** by averaging their logits (weighted by `tied_beta`) and drawing **one** amino acid applied to every member — enforces identical sequences across chains (homooligomers) or correlated positions. |
| `conditional_probs(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False)` | Per-residue `p(s_i | s_rest, backbone)`: for each designed residue, reruns the decoder with only that residue's decoding order moved to the front (or, `backbone_only=True`, with everything else first). |
| `unconditional_probs(X, mask, residue_idx, chain_encoding_all)` | One-pass `p(s_i | backbone)` with **no sequence conditioning** (empty backward mask) — a structure-conditioned per-position "PSSM". |

---

## 5. Inference Entry Point — `protein_mpnn_run.py`

CLI script; all logic lives in `main(args)`.

### 5.1 Flow

1. **Seed handling** — `--seed 0` → random seed; else fixed. Sets torch/numpy/random seeds.
2. **Model selection** — resolves weight folder from `--path_to_model_weights`, else `ca_model_weights/` (`--ca_only`), `soluble_model_weights/` (`--use_soluble_model`), or `vanilla_model_weights/`; loads `<model_name>.pt` checkpoint (`noise_level`, `num_edges` metadata + `model_state_dict`).
3. **Constraint loading** — reads each optional JSONL (`chain_id_jsonl`, `fixed_positions_jsonl`, `pssm_jsonl`, `omit_AA_jsonl`, `bias_AA_jsonl`, `tied_positions_jsonl`, `bias_by_res_jsonl`) if the file exists.
4. **Input resolution** — either `--pdb_path` (single PDB, parsed via `parse_PDB`, wrapped in `StructureDatasetPDB`, and `--pdb_path_chains` builds a `chain_id_dict` on the fly) or `--jsonl_path` (pre-parsed, via `StructureDataset`).
5. **Output folder setup** — creates `seqs/` always; `scores/`, `score_only/`, `conditional_probs_only/`, `unconditional_probs_only/`, `probs/` as flags demand.
6. **Per-protein loop** (`torch.no_grad()`):
   - `tied_featurize(...)` → tensors; `pssm_log_odds_mask = log_odds > pssm_threshold`.
   - **Branch A — `--score_only`**: scores the PDB sequence (and optionally sequences from `--path_to_fasta`, chains sorted alphabetically, `/`-separated); saves npz; prints mean/std for designed residues and global.
   - **Branch B — `--conditional_probs_only`**: calls `model.conditional_probs(...)`, saves `[B,L,21]` log-probs npz.
   - **Branch C — `--unconditional_probs_only`**: calls `model.unconditional_probs(...)` (one forward pass), saves npz.
   - **Branch D — default design**: initial scoring, then for each temperature × NUM_BATCHES: `model.sample(...)` (or `model.tied_sample(...)` when `tied_positions_jsonl` given), re-score samples with `use_input_decoding_order=True` + sampled `decoding_order`, compute per-sample sequence recovery (`one_hot(S)·one_hot(S_sample)` over design mask), append to FASTA. `--save_score`/`--save_probs` dump npz aggregates.

### 5.2 CLI flags (grouped)

| Group | Flags |
|---|---|
| Model | `--path_to_model_weights`, `--model_name` (v_48_002/010/020/030, default v_48_020), `--use_soluble_model`, `--ca_only`, `--backbone_noise` |
| Input | `--pdb_path`, `--pdb_path_chains`, `--jsonl_path`, `--max_length` |
| Constraints | `--chain_id_jsonl`, `--fixed_positions_jsonl`, `--omit_AAs`, `--omit_AA_jsonl`, `--bias_AA_jsonl`, `--bias_by_res_jsonl`, `--pssm_jsonl`, `--pssm_multi`, `--pssm_threshold`, `--pssm_log_odds_flag`, `--pssm_bias_flag`, `--tied_positions_jsonl` |
| Sampling | `--num_seq_per_target`, `--batch_size`, `--sampling_temp` (space-separated list, e.g. `"0.1 0.2"`), `--seed` |
| Modes | `--score_only` (+ `--path_to_fasta`), `--conditional_probs_only` (+ `--conditional_probs_only_backbone`), `--unconditional_probs_only` |
| Output | `--out_folder`, `--save_score`, `--save_probs`, `--suppress_print` |

Note: `NUM_BATCHES = num_seq_per_target // batch_size`; total sequences = `len(temperatures) × NUM_BATCHES × batch_size`.

---

## 6. Training Pipeline — `training/`

### 6.1 `training/training.py` — training loop

1. Builds `params` pointing at the training data folder (`list.csv`, `valid_clusters.txt`, `test_clusters.txt`, per-PDB `.pt` shards).
2. `build_training_clusters(params, debug)` (in `utils.py`) → train/valid/test dicts keyed by cluster id.
3. `PDB_dataset` + `torch.utils.data.DataLoader` (4 workers) wrap raw shard loading; `get_pdbs` runs **asynchronously** via `ProcessPoolExecutor` + queue to prefetch the next epoch's examples while training the current one (data reloaded every `--reload_data_every_n_epochs`).
4. Prefetched dicts → `StructureDataset` → `StructureLoader` (token batching, `batch_size=10000` tokens).
5. Per batch: `featurize(batch, device)` (training variant), forward, `loss_smoothed` (label smoothing) → backprop (optional `torch.cuda.amp` mixed precision + gradient clipping), metrics via `loss_nll` (perplexity + accuracy).
6. Checkpoints saved every epoch as `epoch_last.pt` and every N epochs as `epoch{N}_step{M}.pt`, embedding `num_edges` and `noise_level` so inference can reconstruct the graph config.

Key flags: `--path_for_training_data`, `--path_for_outputs`, `--previous_checkpoint`, `--num_epochs`, `--hidden_dim 128`, `--num_neighbors 48`, `--backbone_noise 0.2`, `--rescut 3.5`, `--mixed_precision`, `--debug`.

### 6.2 `training/model_utils.py` — training-side model utilities

| Item | Description |
|---|---|
| `featurize(batch, device)` | Training featurizer (simpler than `tied_featurize`): no constraints; **randomly shuffles chain order**; promotes visible chains identical to a masked chain into masked (homomer consistency); returns `X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all`. `mask_self` marks intra-chain blocks (kept for interface-loss experiments). |
| `loss_nll` | Same as inference, additionally returns `true_false` (argmax accuracy mask). |
| `loss_smoothed` | Label-smoothed loss; note `loss_av` is normalized by a **fixed 2000.0** (token-normalized across batches) unlike the inference variant. |
| `EncLayer / DecLayer / PositionWiseFeedForward / PositionalEncodings / ProteinFeatures / ProteinMPNN` | Mirror of inference classes. Differences: `ProteinMPNN.forward` takes no `decoding_order` arg (samples `randn` internally); encoder/decoder layers wrapped in `torch.utils.checkpoint.checkpoint` (activation memory saving); default `k_neighbors=32`, `augment_eps=0.1`. |
| `NoamOpt` / `get_std_opt` | Adam with Noam LR warmup schedule: `lr = 2 · d_model^-0.5 · min(step^-0.5, step·4000^-1.5)`, betas (0.9, 0.98). |

### 6.3 `training/utils.py` — training data loading

| Item | Description |
|---|---|
| `StructureDataset` / `StructureLoader` | Same concepts as core library (loader here yields index lists internally). |
| `NoamOpt` / `get_std_opt` | Duplicate of scheduler (used by `training.py` import from `model_utils`, kept here historically). |
| `worker_init_fn`, `get_pdbs` | DataLoader worker seeding; iterates a loader for `num_units` examples, converting raw shards into parsed chain dicts (respects `max_length`). |
| `PDB_dataset` | `torch.utils.data.Dataset`: picks a **random chain/assembly** from the cluster each `__getitem__`. |
| `loader_pdb(item, params)` | Loads `PDBID.pt` metadata + chain shards; randomly selects a **biological assembly** containing the chain, applies stored 4×4 transforms (`asmb_xformIDX`) to generate multi-chain complexes; chains with >70% seq-id to the target chain are marked masked (so homomers are designed with tied-ish supervision). |
| `build_training_clusters(params, debug)` | Parses `list.csv`, filters by resolution ≤ `RESCUT` and deposition ≤ `DATCUT`, splits rows into train/valid/test by cluster id from `valid_clusters.txt`/`test_clusters.txt`. |

### 6.4 `training/parse_cif_noX.py` — dataset preparation

Offline tool to build the `.pt` shards from mmCIF files: `parse_mmcif(filename)` (chain extraction incl. HETATM=MSE handling), `parseAssemblies` (PDB biounit transform expressions via `parseOperationExpression`), `TMalign` wrapper (calls external TM-align binary; `get_tm_pairs` builds pairwise similarity matrices), `writepdb` (re-emits PDB for debugging).

### 6.5 Training data format (from `training/README.md`)

Download: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz` (16.5 GB) or `.../pdb_2021aug02_sample.tar.gz` (47 MB test subsample).

- `PDBID_CHAINID.pt` — `seq`, `xyz [L,14,3]`, `mask [L,14]`, `bfac [L,14]`, `occ [L,14]`
- `PDBID.pt` — metadata: `method/date/resolution/chains/tm (TM-align pairwise)/asmb_ids/asmb_chains/asmb_xformIDX ...`
- `list.csv` — `CHAINID, DEPOSITION, RESOLUTION, HASH, CLUSTER, SEQUENCE`
- `valid_clusters.txt` / `test_clusters.txt` — held-out cluster ids

---

## 7. Helper Scripts — `helper_scripts/`

Standalone `argparse` scripts that produce the JSONL dictionaries consumed by `protein_mpnn_run.py`. All share the pattern: read `parsed_pdbs.jsonl` (+ args) → write one JSON dict per line, keyed by PDB name.

| Script | Output dict shape | Notes |
|---|---|---|
| `parse_multiple_chains.py` | parsed-PDB dicts (§4.1) | `--input_path` folder of PDBs, `--output_path`, `--ca_only`. **Mandatory first step** for the jsonl workflow. |
| `assign_fixed_chains.py` | `{"<pdb>": [["A","B"], ["C"]]}` (designed, fixed) | `--chain_list` = chains **to design**; blank = design all. |
| `make_fixed_positions_dict.py` | `{"<pdb>": {"A": [[1,2,3]]}}` | `--chain_list`, `--position_list` (1-based, comma = chain groups); `--specify_non_fixed` inverts to "only design these". |
| `make_tied_positions_dict.py` | `{"<pdb>": [{"A": [[1,2,3]], "B": [[1,2,3]]}]}` | `--homooligomer 1` auto-derives symmetric ties from identical chain sequences. |
| `make_pos_neg_tied_positions_dict.py` | tied positions + `tied_beta` weights | Positive chains (β=1.0) designed to match; negative chains (β=−0.1/−0.5) pushed away. |
| `make_bias_AA.py` | `{"A": -1.1, "F": 0.7}` | Global AA logit bias. |
| `make_bias_per_res_dict.py` | `{"<pdb>": {"A": {"bias": [[...21 floats...]]}}}` | Per-residue bias (aligned to parsed sequences). |
| `make_pssm_input_dict.py` | PSSM coef/bias/log_odds per chain | From npz PSSM profiles + parsed jsonl. |
| `other_tools/make_pssm_dict.py` | PSSM dict from raw PSSM files | |
| `other_tools/make_omit_AA.py` | per-position omitted AAs | Feeds `--omit_AA_jsonl`. |

---

## 8. Dependency Relationships

### 8.1 Internal imports

```
protein_mpnn_run.py ─────────────► protein_mpnn_utils
    │   (loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t,
    │    cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize,
    │    parse_PDB, parse_fasta, StructureDataset, StructureDatasetPDB, ProteinMPNN)
    │
training/training.py ─┬─► training/utils.py      (worker_init_fn, get_pdbs, loader_pdb,
    │                 │                           build_training_clusters, PDB_dataset,
    │                 │                           StructureDataset, StructureLoader)
    │                 └─► training/model_utils.py (featurize, loss_smoothed, loss_nll,
    │                                               get_std_opt, ProteinMPNN)
    │
helper_scripts/*.py  (self-contained; only argparse/numpy/json)
training/parse_cif_noX.py (self-contained; shells out to TM-align binary)
```

Note: the model & NN-building-block classes exist in **two near-duplicate copies** — `protein_mpnn_utils.py` (inference features: decoding-order control, tied sampling, conditional/unconditional probs, CA-only) and `training/model_utils.py` (gradient checkpointing, training featurize/losses). Keep this in mind when patching the architecture.

### 8.2 External dependencies

| Package | Usage |
|---|---|
| `torch` | Model, tensors, AMP, checkpointing, optim |
| `numpy` | Featurization, parsing, stats |
| Python stdlib | `json`, `argparse`, `glob`, `subprocess` (git hash in FASTA headers), `concurrent.futures`, `queue`, `itertools`, `csv`, `dateutil` (training only) |
| TM-align binary | Only for `training/parse_cif_noX.py` dataset prep |

No `requirements.txt` is pinned; README suggests conda + PyTorch (e.g. cudatoolkit 11.3).

### 8.3 Data dependencies

```
vanilla/soluble/ca_model_weights/*.pt ──► protein_mpnn_run.py
helper_scripts outputs (*.jsonl)      ──► protein_mpnn_run.py constraint flags
inputs/*/pdbs/*.pdb                   ──► helper_scripts/parse_multiple_chains.py
pdb_2021aug02/ training set           ──► training/training.py
```

---

## 9. Model Weights

| Folder | Models | Trained on | Use |
|---|---|---|---|
| `vanilla_model_weights/` | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` | Full PDB biounits | Default (name = 48 neighbors + training noise ×0.01). **v_48_020 = default**, generally best. |
| `soluble_model_weights/` | `v_48_002/010/020/030.pt` (+`excluded_PDBs.csv`) | Soluble proteins only | Better for soluble/secreted designs. |
| `ca_model_weights/` | `v_48_002/010/020.pt` | CA-only backbone | Use with `--ca_only` when only CA traces exist (lower accuracy). |

Checkpoint contents: `{'model_state_dict', 'optimizer_state_dict', 'epoch', 'step', 'num_edges' (=k_neighbors), 'noise_level'}`. At inference `num_edges` overrides the constructor default, so the model name's "48" is preserved automatically.

---

## 10. Running the Project

### 10.1 Environment

```bash
conda create --name mlfold
conda activate mlfold
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install numpy
# Python >= 3.0; CPU works, CUDA GPU recommended
```

### 10.2 Quick start — design sequences for one PDB

```bash
python protein_mpnn_run.py \
        --pdb_path inputs/PDB_monomers/pdbs/5L33.pdb \
        --out_folder outputs/demo \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
```

### 10.3 Multi-PDB jsonl workflow (canonical pipeline)

```bash
# 1) Parse a folder of PDBs
python helper_scripts/parse_multiple_chains.py \
        --input_path inputs/PDB_complexes/pdbs/ \
        --output_path outputs/parsed_pdbs.jsonl

# 2) (optional) Build constraint dicts, e.g. fixed positions
python helper_scripts/make_fixed_positions_dict.py \
        --input_path outputs/parsed_pdbs.jsonl \
        --output_path outputs/fixed_positions.jsonl \
        --chain_list "A B" --position_list "1 2 3 4, 5 6 7"

# 3) Design
python protein_mpnn_run.py \
        --jsonl_path outputs/parsed_pdbs.jsonl \
        --chain_id_jsonl outputs/chains.jsonl \
        --fixed_positions_jsonl outputs/fixed_positions.jsonl \
        --out_folder outputs/run1 \
        --num_seq_per_target 4 --sampling_temp "0.1 0.2" --batch_size 2
```

More scenarios (homooligomer tying, AA bias, PSSM, scoring, conditional/unconditional probs): see `examples/submit_example_*.sh`, each of which is a runnable SLURM script.

### 10.4 Score a known backbone–sequence pair

```bash
python protein_mpnn_run.py \
        --pdb_path my.pdb --score_only 1 --out_folder outputs/score
# optionally add --path_to_fasta seqs.fa  (chains alphabetical, '/'-separated)
```

### 10.5 Colab

`colab_notebooks/quickdemo.ipynb` (interactive design + AlphaFold2 validation), `ca_only_quickdemo.ipynb`, `quickdemo_wAF2.ipynb`.

### 10.6 Retraining

```bash
# 1) Download data (see §6.5) and unpack to e.g. ./pdb_2021aug02
# 2) (optional) Regenerate shards from mmCIF with training/parse_cif_noX.py
# 3) Train
cd training
python training.py \
        --path_for_training_data ../pdb_2021aug02 \
        --path_for_outputs ./exp_020 \
        --num_epochs 150 --num_neighbors 48 --backbone_noise 0.20
# quick smoke test: add --debug True
```

Training outputs `log.txt` (perplexity/accuracy per epoch) and `model_weights/epoch*.pt` checkpoints directly loadable by `protein_mpnn_run.py --path_to_model_weights`.

### 10.7 Windows note (this workspace)

The bundled `examples/*.sh` are bash/SLURM scripts; on Windows run the equivalent `python ...` commands directly (PowerShell line continuation uses backticks), or use the Colab notebooks. CUDA optional — the code auto-falls back to CPU (`torch.cuda.is_available()` check in both entry points).

---

## 11. Output Format Reference

`outputs/<run>/seqs/<name>.fa` — one record per generated sequence:

```
>3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], designed_chains=['A', 'C'],
model_name=v_48_020, git_hash=<commit>, seed=37
<Native sequence, chains joined by '/'>
>T=0.1, sample=1, score=0.7291, global_score=0.9330, seq_recovery=0.5736
<Sampled sequence, chains joined by '/'>
```

- `score` — mean NLL over **designed** residues only; `global_score` — over all residues.
- `seq_recovery` — fraction of designed positions matching the native AA (sanity metric, not a quality goal).

Additional npz outputs by flag: `scores/*.npz` (`score`, `global_score`), `probs/*.npz` (`probs`, `log_probs`, `S`, `mask`, `chain_order`), `score_only/*.npz`, `conditional_probs_only/*.npz`, `unconditional_probs_only/*.npz` (`log_p [B,L,21]`, `S`, `mask`, `design_mask`).

---

## 12. Key Domain Concepts

- **Alphabet** — `'ACDEFGHIKLMNPQRSTVWYX'`: 20 standard AAs + `X` (unknown/gap); all tensors use 21 classes.
- **Masked vs. visible chains** — masked = to be designed (`chain_M=1`), visible = kept fixed as context (ligand-binding partner chains, etc.).
- **`chain_M` vs. `chain_M_pos`** — chain-level vs. residue-level designability; the model uses their product `chain_M * chain_M_pos` as the final design mask.
- **Decoding order** — random permutation of designed residues (weighted by design mask); sampling is autoregressive *along this order*, and re-scoring reuses the same order (`use_input_decoding_order=True`).
- **Tied positions** — groups sampled as one AA (symmetry design); betas weight each member's contribution to the shared logits.
- **PSSM integration** — two mechanisms: probability mixing (`pssm_multi`) and log-odds gating (`pssm_threshold` + masking), both per-residue × per-AA.
- **kNN graph (`E_idx`)** — sparse 48-neighbor graph; all attention/message passing is restricted to it, giving linear-in-L memory vs. quadratic full attention.
- **residue_idx chain jump** — position indices encode `100*(chain-1) + i` so relative positional encodings never bridge separate chains.
