# ======================================================================================
# Quantifying the qualitative Jaccard-neighbour result (Fig. 12, right panel).
#
# The grids in `plot_similarity_grids.py` show, for a fixed query image, its nearest
# neighbours under the Jaccard similarity of SAE feature-activation patterns. The claim
# is that the LCH (centroid) dictionary retrieves neighbours in a consistent spatial
# configuration ("a dog with the same head position") more reliably than the LRH (latent)
# dictionary. This script replaces the eye-test with a measurement: it runs the same
# retrieval for many queries and scores each retrieved neighbour with an independent judge.
#
# The judge is DINOv3, which shares no weights with the DINOv2 MLP block the dictionaries
# decompose -- scoring with DINOv2 itself would be circular. Its patch tokens are compared
# position-wise rather than pooled into one vector per image, because pooled embeddings
# measure "same kind of object" and empirically cannot separate the two dictionaries,
# while position-wise agreement is what the spatial-configuration claim is about.
#
# Reported per k: mean judge similarity between query and its top-k neighbours (+/- s.e.m.
# over queries), the fraction of neighbours sharing the query's breed, and a paired
# Wilcoxon test. Averaged over several SAE training seeds, with the between-seed spread
# reported separately so the effect can be checked against retraining variance.
#
# Prerequisites (produced by dino-dogs/extraction.py and dino-dogs/training.py):
#   outputs/dino-dogs/v2/{split}-cls/{centroids,latents,labels}.pt
#   outputs/dino-dogs/v2/v2-cls-{centroids,latents}-10exp-32K-{seed}.pt
# ======================================================================================
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPImageProcessor


# ======================================================================================
# 1. SAE (must match dino-dogs/training.py)
# ======================================================================================
def topk_sparsify(x: torch.Tensor, k: int) -> torch.Tensor:
    with torch.no_grad():
        _, indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
    return x * mask


class TopKAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, topk: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.topk = topk

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = topk_sparsify(F.relu(self.encoder(x)), self.topk)
        return self.decoder(z), z


# ======================================================================================
# 2. Jaccard retrieval
# ======================================================================================
@torch.no_grad()
def binary_activation_matrix(
    sae: TopKAutoencoder, features: torch.Tensor, device: torch.device, batch_size: int = 1024
) -> torch.Tensor:
    """(N, hidden) float matrix with 1.0 wherever the SAE feature fires."""
    sae.eval()
    rows = []
    for start in tqdm(range(0, features.size(0), batch_size), desc="SAE encode", leave=False):
        _, z = sae(features[start : start + batch_size].to(device))
        rows.append((z > 1e-8).float())
    return torch.cat(rows)


@torch.no_grad()
def topk_jaccard_neighbours(
    binary: torch.Tensor, queries: torch.Tensor, k: int, tie_break: torch.Tensor, block: int = 512
) -> torch.Tensor:
    """
    (len(queries), k) indices of each query row's k most Jaccard-similar other rows.

    Only the query rows are scored, but every row of `binary` remains a candidate: the
    retrieval pool is always the full split, so subsampling queries cheapens the analysis
    without making the retrieval task easier.

    Ties are common when the codes are short, and the dataset is stored in class order,
    so a stable argsort would silently prefer low indices (i.e. the query's own class).
    `tie_break` is a fixed random tensor added at a magnitude far below the smallest
    possible Jaccard gap, which randomises tied orderings without perturbing real ones.
    """
    sizes = binary.sum(dim=1)  # |A| per row
    out = torch.empty(queries.numel(), k, dtype=torch.long, device=binary.device)

    for start in tqdm(range(0, queries.numel(), block), desc="Jaccard retrieval", leave=False):
        stop = min(start + block, queries.numel())
        rows = queries[start:stop]
        inter = binary[rows] @ binary.T
        union = sizes[rows, None] + sizes[None, :] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        sim = sim + tie_break[None, :]
        sim[torch.arange(stop - start, device=binary.device), rows] = -1.0  # drop self
        out[start:stop] = sim.topk(k, dim=1).indices

    return out


# ======================================================================================
# 3. Independent judges
#
# The judge scores how alike a query and a retrieved neighbour are. It must be a model
# the dictionaries were not built from, or the comparison is circular -- the dictionaries
# here decompose a DINOv2 MLP block, so DINOv2 is excluded and DINOv3 (a different
# training run and architecture generation) and CLIP are both admissible.
#
# Two judge modes, because they answer different questions:
#   global  -- one pooled embedding per image. Measures "same kind of thing". CLIP's
#              pooled embedding in particular is trained for caption-level alignment and
#              is largely pose-invariant by construction.
#   spatial -- per-patch tokens compared at matched grid positions, which stays sensitive
#              to layout and pose. This is the mode that can test the paper's actual
#              sentence ("a dog with the same head position"); a pooled embedding cannot.
# ======================================================================================
JUDGES = {
    "clip": "openai/clip-vit-base-patch16",
    # DINOv3 is a gated repo -- needs `huggingface-cli login` with the licence accepted.
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    # DINOv1 is an ungated stand-in: self-supervised like DINOv2 but a separate model and
    # training recipe, so it is still an independent judge for a DINOv2-derived dictionary.
    "dinov1": "facebook/dino-vitb16",
}


def get_image_subset(data_root: str, split: str) -> Subset:
    """The same 10-dog-breed subset used by extraction.py, kept as PIL images."""
    dataset = torchvision.datasets.ImageNet(data_root, split=split, transform=None)
    random.seed(0)
    selected_classes = set(random.sample(list(range(151, 269)), 10))
    indices = [i for i, target in enumerate(dataset.targets) if target in selected_classes]
    return Subset(dataset, indices)


@torch.no_grad()
def judge_embeddings(
    dataset: Subset,
    needed: torch.Tensor,
    device: torch.device,
    batch_size: int,
    judge: str,
    judge_name: str,
    mode: str,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    L2-normalised judge embeddings for just the images in `needed`.

    Reading images off disk dominates the runtime, so we embed only the images the
    analysis actually touches (queries, their retrieved neighbours, and the random
    baseline draws) rather than the whole split. Returns the embedding matrix together
    with a lookup from dataset index to row in that matrix.

    In `spatial` mode each patch token is normalised individually and the grid is then
    flattened, so a dot product between two rows equals the mean per-position cosine
    similarity -- i.e. agreement of content at matching image locations.
    """
    order = sorted(int(i) for i in needed.tolist())
    lookup = {idx: row for row, idx in enumerate(order)}
    print(f"Embedding {len(order)} of {len(dataset)} images with {judge_name} ({mode}) ...")

    # use_safetensors: transformers refuses .bin checkpoints on torch < 2.6 (CVE-2025-32434).
    if judge == "clip":
        model = CLIPModel.from_pretrained(judge_name, use_safetensors=True).to(device).eval()
        processor = CLIPImageProcessor.from_pretrained(judge_name)
    else:
        model = AutoModel.from_pretrained(judge_name, use_safetensors=True).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(judge_name)

    def collate(batch):
        images = [img.convert("RGB") for img, _ in batch]
        return processor(images=images, return_tensors="pt")["pixel_values"]

    loader = DataLoader(
        Subset(dataset, order), batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=collate
    )

    def features(pixel_values: torch.Tensor) -> torch.Tensor:
        if judge == "clip":
            out = model.get_image_features(pixel_values=pixel_values)
            # transformers >=5 returns a model output whose pooler_output holds the
            # projected (joint-space) embedding; older versions return it directly.
            pooled = out if isinstance(out, torch.Tensor) else out.pooler_output
            if mode == "global":
                return F.normalize(pooled, dim=-1)
            tokens = model.vision_model(pixel_values=pixel_values).last_hidden_state
        else:
            out = model(pixel_values=pixel_values)
            if mode == "global":
                return F.normalize(out.last_hidden_state[:, 0], dim=-1)  # CLS token
            tokens = out.last_hidden_state

        # Patch tokens are the trailing tokens: DINOv3 prefixes CLS + register tokens and
        # CLIP prefixes CLS, so counting back from the end is robust to both.
        config = getattr(model.config, "vision_config", model.config)  # CLIP nests its config
        side = pixel_values.shape[-1] // config.patch_size
        patches = tokens[:, -(side * side) :, :]
        # Normalise per position, then scale so a dot product averages over positions.
        return (F.normalize(patches, dim=-1) / np.sqrt(side * side)).flatten(1)

    dtype = torch.float32 if mode == "global" else torch.float16
    embeddings = [
        features(px.to(device)).to(dtype).cpu() for px in tqdm(loader, desc=judge, leave=False)
    ]
    del model
    torch.cuda.empty_cache()
    return torch.cat(embeddings), lookup


def to_rows(indices: torch.Tensor, lookup: Dict[int, int]) -> torch.Tensor:
    """Remap dataset indices onto rows of the (sub-sampled) judge embedding matrix."""
    return torch.tensor([[lookup[int(i)] for i in row] for row in indices.reshape(indices.size(0), -1)])


# ======================================================================================
# 4. Scoring
# ======================================================================================
def coherence_scores(
    queries: torch.Tensor,
    neighbours: torch.Tensor,
    embeddings: torch.Tensor,
    lookup: Dict[int, int],
    labels: torch.Tensor,
    k: int,
) -> Dict[str, np.ndarray]:
    """Per-query mean judge cosine similarity to the top-k neighbours, and breed purity.

    `queries` and `neighbours` hold dataset indices; `lookup` maps those onto rows of the
    sub-sampled embedding matrix.
    """
    idx = neighbours[:, :k]
    query_emb = embeddings[to_rows(queries.unsqueeze(1), lookup).squeeze(1)][:, None, :]
    # float32 for the reduction: spatial embeddings are stored as fp16 and the dot product
    # runs over ~150k terms, where fp16 accumulation would lose the precision we need.
    cos = (query_emb.float() * embeddings[to_rows(idx, lookup)].float()).sum(dim=-1)  # (Q, k)
    same_breed = (labels[idx] == labels[queries][:, None]).float()
    return {"judge_cosine": cos.mean(dim=1).numpy(), "breed_purity": same_breed.mean(dim=1).numpy()}


def random_neighbours(n: int, n_queries: int, k_max: int, seed: int) -> torch.Tensor:
    """Uniform draws from the full retrieval pool -- the semantic floor for this metric."""
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randint(0, n, (n_queries, k_max), generator=generator)


def summarise(name: str, scores: np.ndarray) -> Dict[str, float]:
    return {
        "method": name,
        "mean": float(scores.mean()),
        "sem": float(scores.std(ddof=1) / np.sqrt(len(scores))),
        "n": int(len(scores)),
    }


# ======================================================================================
# 5. Main
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Judge-scored coherence of Jaccard neighbours.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dino-dogs"))
    parser.add_argument("--data_root", type=str, default="/mnt/richb/datasets/ImageNet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model_type", type=str, default="v2")
    parser.add_argument("--tokens", type=str, default="cls")
    parser.add_argument("--expansion", type=int, default=10)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="Analysis seed: query sampling and tie-breaks.")
    parser.add_argument(
        "--sae_seeds",
        type=int,
        nargs="+",
        default=[0],
        help="SAE training seeds to average over. Held separate from --seed so the query set is "
        "identical across seeds and the LCH/LRH comparison stays paired.",
    )
    parser.add_argument("--k_max", type=int, default=8, help="Largest k in the reported sweep.")
    parser.add_argument(
        "--include_random",
        action="store_true",
        help="Also score uniformly random neighbours, giving the floor for the metric. Off by "
        "default because the random draws dominate the set of images that must be embedded "
        "(roughly tripling runtime); the reported LCH/LRH comparison does not depend on it.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="dinov3",
        choices=sorted(JUDGES),
        help="Scoring model. DINOv2 is deliberately unavailable -- it is what the dictionaries decompose.",
    )
    parser.add_argument("--judge_name", type=str, default=None, help="Override the judge checkpoint.")
    parser.add_argument(
        "--judge_mode",
        type=str,
        default="spatial",
        choices=["global", "spatial"],
        help="'spatial' compares patch tokens position-wise (the reported setting); 'global' pools "
        "one embedding per image, which does not separate the two dictionaries.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--n_queries",
        type=int,
        default=1000,
        help="Queries to score (<=0 uses every image). The retrieval pool is always the full split.",
    )
    args = parser.parse_args()
    judge_name = args.judge_name or JUDGES[args.judge]
    seed_tag = "seed" + "".join(str(s) for s in args.sae_seeds)
    tag = f"{args.split}-{args.judge}-{args.judge_mode}-{args.topk}K-{seed_tag}"

    device = torch.device(args.device)
    run_dir = args.output_dir / args.model_type
    feature_dir = run_dir / f"{args.split}-{args.tokens}"

    labels = torch.load(feature_dir / "labels.pt", weights_only=True)
    n = labels.size(0)

    image_split = "train" if args.split == "train" else "val"
    images = get_image_subset(args.data_root, image_split)
    assert len(images) == n, f"{len(images)} images vs {n} feature rows -- subsets disagree."

    # --- Choose queries (retrieval still ranks against all n images) ---
    generator = torch.Generator().manual_seed(args.seed)
    if args.n_queries <= 0 or args.n_queries >= n:
        queries = torch.arange(n)
    else:
        queries = torch.randperm(n, generator=generator)[: args.n_queries].sort().values
    print(f"Scoring {queries.numel()} queries against a pool of {n} images ({args.split} split).")

    # --- Retrieve neighbours under each dictionary ---
    tie_break = (
        torch.rand(n, generator=torch.Generator().manual_seed(args.seed)).to(device) * 1e-6
    )
    # Keyed (sae_seed, activation_type); the random floor does not depend on the SAE.
    neighbours: Dict[Tuple[int, str], torch.Tensor] = {}
    for sae_seed in args.sae_seeds:
        for activation_type in ["centroids", "latents"]:
            run_name = (
                f"{args.model_type}-{args.tokens}-{activation_type}-"
                f"{args.expansion}exp-{args.topk}K-{sae_seed}"
            )
            features = torch.load(feature_dir / f"{activation_type}.pt", weights_only=True)
            sae = TopKAutoencoder(features.size(1), features.size(1) * args.expansion, args.topk)
            sae.load_state_dict(
                torch.load(run_dir / f"{run_name}.pt", map_location=device, weights_only=True)
            )
            sae.to(device)

            print(f"--- seed {sae_seed}, {activation_type} ---")
            binary = binary_activation_matrix(sae, features, device)
            neighbours[(sae_seed, activation_type)] = topk_jaccard_neighbours(
                binary, queries.to(device), args.k_max, tie_break
            ).cpu()
            del binary, features, sae
            torch.cuda.empty_cache()
    random_idx = (
        random_neighbours(n, queries.numel(), args.k_max, args.seed) if args.include_random else None
    )

    # --- Embed only the images the analysis touches (once, across all seeds) ---
    touched = [queries] + [idx.flatten() for idx in neighbours.values()]
    if random_idx is not None:
        touched.append(random_idx.flatten())
    needed = torch.cat(touched).unique()
    embeddings, lookup = judge_embeddings(
        images, needed, device, args.batch_size, args.judge, judge_name, args.judge_mode
    )

    # --- Score ---
    # Two sources of variation are reported separately: `sem` is the s.e.m. over queries
    # (pooling seeds), while `seed_std` is the spread of the per-seed means -- the latter
    # is what shows whether the LCH/LRH ordering survives SAE retraining.
    METHODS = [("LCH (centroids)", "centroids"), ("LRH (latents)", "latents")]
    rows: List[Dict] = []
    for k in range(1, args.k_max + 1):
        per_method: Dict[str, Dict[str, np.ndarray]] = {}
        seed_means: Dict[str, List[float]] = {}
        for name, key in METHODS:
            by_seed = [
                coherence_scores(queries, neighbours[(s, key)], embeddings, lookup, labels, k)
                for s in args.sae_seeds
            ]
            seed_means[name] = [float(scores["judge_cosine"].mean()) for scores in by_seed]
            # Average each query's score over seeds, so the paired test is over queries.
            per_method[name] = {
                field: np.mean([scores[field] for scores in by_seed], axis=0)
                for field in ("judge_cosine", "breed_purity")
            }
        if random_idx is not None:
            per_method["Random"] = coherence_scores(queries, random_idx, embeddings, lookup, labels, k)
            seed_means["Random"] = [float(per_method["Random"]["judge_cosine"].mean())]

        _, pvalue = wilcoxon(
            per_method["LCH (centroids)"]["judge_cosine"], per_method["LRH (latents)"]["judge_cosine"]
        )
        # Does LCH win on every individual seed, or only on average?
        wins = sum(
            lch > lrh for lch, lrh in zip(seed_means["LCH (centroids)"], seed_means["LRH (latents)"])
        )
        for name in per_method:
            means = seed_means[name]
            rows.append(
                {
                    "k": k,
                    **summarise(name, per_method[name]["judge_cosine"]),
                    "breed_purity": float(per_method[name]["breed_purity"].mean()),
                    "wilcoxon_p": float(pvalue),
                    "seed_means": means,
                    "seed_std": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                    "lch_seed_wins": f"{wins}/{len(args.sae_seeds)}",
                }
            )

    # --- Report ---
    print(
        f"\n{judge_name} ({args.judge_mode}) coherence of top-k Jaccard neighbours "
        f"({args.split} split, {queries.numel()} queries, pool={n}, SAE seeds={args.sae_seeds})"
    )
    print(
        f"{'k':>3} {'method':<18} {'judge cos':>16} {'seed sd':>9} {'breed purity':>13}"
        f" {'paired p':>10} {'LCH wins':>9}"
    )
    for row in rows:
        print(
            f"{row['k']:>3} {row['method']:<18} {row['mean']:>8.4f} +/- {row['sem']:.4f}"
            f" {row['seed_std']:>9.4f} {row['breed_purity']:>13.3f} {row['wilcoxon_p']:>10.2e}"
            f" {row['lch_seed_wins']:>9}"
        )

    results_path = args.output_dir / f"neighbor_coherence-{tag}.json"
    results_path.write_text(
        json.dumps(
            {"config": vars(args) | {"output_dir": str(args.output_dir), "judge_name": judge_name}, "rows": rows},
            indent=2,
            default=str,
        )
    )
    print(f"\nResults written to {results_path}")

    # --- Plot (the two dictionaries only; the random floor is far below and would flatten
    # the scale that the LCH/LRH separation lives on) ---
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    styles = {
        "LCH (centroids)": dict(color=plt.cm.winter(0.25), marker="o", label="LCH"),
        "LRH (latents)": dict(color=plt.cm.autumn(0.25), marker="s", linestyle="--", label="LRH"),
    }
    for name, style in styles.items():
        series = [row for row in rows if row["method"] == name]
        ks = [row["k"] for row in series]
        means = np.array([row["mean"] for row in series])
        sems = np.array([row["sem"] for row in series])
        ax.plot(ks, means, **style)
        ax.fill_between(ks, means - sems, means + sems, color=style["color"], alpha=0.2)
    ax.set_xlabel("Number of Jaccard neighbours $k$")
    ax.set_ylabel("Mean DINOv3 spatial similarity to query")
    ax.grid(True, linestyle="dashed", color="gray", alpha=0.3)
    ax.legend()
    plot_path = args.output_dir / f"neighbor_coherence-{tag}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
