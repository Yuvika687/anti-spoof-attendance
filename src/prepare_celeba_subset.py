"""
prepare_celeba_subset.py
-------------------------
Builds a manageable, identity-disjoint subset of CelebA-Spoof for training
the liveness classifier, instead of loading all ~625k images.

Expected RAW CelebA-Spoof layout (as distributed officially):

    <raw_root>/
        Data/
            train/
                <identity_id>/
                    live/  *.jpg | *.png
                    spoof/ *.jpg | *.png
            test/
                <identity_id>/
                    live/
                    spoof/

Each <identity_id> folder belongs to a single person, which is exactly what
we need for an IDENTITY-DISJOINT split: we assign whole identity folders to
train / val / test so no person appears in more than one split.

Output layout (consumed by dataset.py):

    data/celeba_spoof/
        train/{live,spoof}/*.jpg
        val/{live,spoof}/*.jpg
        test/{live,spoof}/*.jpg

Usage:
    python prepare_celeba_subset.py \\
        --raw_root /path/to/CelebA_Spoof \\
        --out_root ../data/celeba_spoof \\
        --live_total 12000 --spoof_total 12000 \\
        --train_frac 0.7 --val_frac 0.15 --test_frac 0.15 \\
        --copy_mode symlink
"""

import argparse
import os
import random
import shutil
from collections import defaultdict

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def scan_identities(raw_root: str):
    """
    Walk <raw_root>/Data/{train,test}/<identity>/{live,spoof}/* and return a
    dict: identity_key -> {"live": [paths], "spoof": [paths]}

    identity_key includes the source split (train/test) prefix so ids don't
    collide across CelebA-Spoof's own train/test partition.
    """
    identities = defaultdict(lambda: {"live": [], "spoof": []})

    data_root = os.path.join(raw_root, "Data")
    if not os.path.isdir(data_root):
        # fall back: allow raw_root to already point at "Data"
        data_root = raw_root

    for source_split in ("train", "test"):
        split_dir = os.path.join(data_root, source_split)
        if not os.path.isdir(split_dir):
            continue
        for identity in sorted(os.listdir(split_dir)):
            identity_dir = os.path.join(split_dir, identity)
            if not os.path.isdir(identity_dir):
                continue
            key = f"{source_split}_{identity}"
            for class_name in ("live", "spoof"):
                class_dir = os.path.join(identity_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(VALID_EXTENSIONS):
                        identities[key][class_name].append(os.path.join(class_dir, fname))

    return identities


def split_identities(identity_keys, train_frac, val_frac, test_frac, seed):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1.0"
    rng = random.Random(seed)
    keys = list(identity_keys)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(keys[:n_train])
    val_ids = set(keys[n_train:n_train + n_val])
    test_ids = set(keys[n_train + n_val:])
    return train_ids, val_ids, test_ids


def collect_and_cap(identities, id_subset, class_name, cap):
    """Gather all image paths for a class across a set of identities, then
    cap the TOTAL count while sampling roughly evenly across identities so
    a handful of people don't dominate the subset.

    Returns a list of (identity_key, filepath) tuples. identity_key is the
    "<source_split>_<identity>" key used during scanning/splitting -- we
    carry it through (rather than re-deriving a name from the file path)
    because CelebA-Spoof's raw train/ and test/ pools can reuse the same
    identity folder name (e.g. both have an "id_4"), which would otherwise
    collide once flattened into a single output folder.
    """
    per_identity = {k: identities[k][class_name] for k in id_subset if identities[k][class_name]}
    if not per_identity:
        return []

    rng = random.Random(0)
    for k in per_identity:
        rng.shuffle(per_identity[k])

    selected = []
    # Round-robin across identities until cap reached or exhausted.
    exhausted = False
    idx = 0
    keys = list(per_identity.keys())
    while len(selected) < cap and not exhausted:
        exhausted = True
        for k in keys:
            if idx < len(per_identity[k]):
                selected.append((k, per_identity[k][idx]))
                exhausted = False
                if len(selected) >= cap:
                    break
        idx += 1
    return selected


def place_files(items, out_dir, copy_mode):
    """items: list of (identity_key, filepath) tuples, as returned by collect_and_cap."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (identity_key, src_path) in enumerate(items):
        # Flatten name using the full identity_key (source_split + identity)
        # to guarantee uniqueness across CelebA-Spoof's own train/test pools.
        ext = os.path.splitext(src_path)[1]
        dst_name = f"{identity_key}_{i:06d}{ext}"
        dst_path = os.path.join(out_dir, dst_name)

        if copy_mode == "symlink":
            if os.path.islink(dst_path) or os.path.exists(dst_path):
                continue
            os.symlink(os.path.abspath(src_path), dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Build identity-disjoint CelebA-Spoof subset")
    parser.add_argument("--raw_root", required=True, help="Path to the raw CelebA-Spoof download")
    parser.add_argument("--out_root", default="../data/celeba_spoof", help="Output subset root")
    parser.add_argument("--live_total", type=int, default=13000)
    parser.add_argument("--spoof_total", type=int, default=13000)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy_mode", choices=["copy", "symlink"], default="symlink",
                         help="'symlink' saves disk space; use 'copy' if you need standalone files")
    args = parser.parse_args()

    print(f"Scanning raw dataset at: {args.raw_root} ...")
    identities = scan_identities(args.raw_root)
    print(f"Found {len(identities)} identity folders.")
    if len(identities) == 0:
        raise SystemExit(
            "No identities found. Check --raw_root points at the CelebA-Spoof "
            "download containing a 'Data/train' and/or 'Data/test' folder."
        )

    train_ids, val_ids, test_ids = split_identities(
        identities.keys(), args.train_frac, args.val_frac, args.test_frac, args.seed
    )
    print(f"Identity split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    # Distribute the total image budget across splits proportional to fractions.
    split_defs = [
        ("train", train_ids, args.train_frac),
        ("val", val_ids, args.val_frac),
        ("test", test_ids, args.test_frac),
    ]

    for split_name, id_subset, frac in split_defs:
        live_cap = max(1, int(args.live_total * frac))
        spoof_cap = max(1, int(args.spoof_total * frac))

        live_files = collect_and_cap(identities, id_subset, "live", live_cap)
        spoof_files = collect_and_cap(identities, id_subset, "spoof", spoof_cap)

        place_files(live_files, os.path.join(args.out_root, split_name, "live"), args.copy_mode)
        place_files(spoof_files, os.path.join(args.out_root, split_name, "spoof"), args.copy_mode)

        print(f"[{split_name}] live={len(live_files)}  spoof={len(spoof_files)} "
              f"-> {os.path.join(args.out_root, split_name)}")

    print("\nDone. Subset written to:", os.path.abspath(args.out_root))


if __name__ == "__main__":
    main()
