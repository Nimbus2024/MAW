"""数据层：forget 用同题配对(idk/forget) recipe；retain 用 plain 配对（按 forget 数重采样）。

- forget: `PairedDataset(dpo=True)` + `collate_dpo` → 每 step `{"mm": {batch_w, batch_l}, "um": {...}}`
  `batch_l` = 原答案(forget)，`batch_w` = 拒绝回答(idk)。**w 是否被前向由 objective/控制器决定。**
- retain: `PairedDataset` + `collate_plain`，每 epoch 抽与 forget 等量的子集。
"""
import os

import pandas as pd
from torch.utils.data import DataLoader, RandomSampler

from ._common import load_idk
from ._paired import PairedDataset, build_pairs, collate_dpo, collate_plain


def _parquets(data_split_dir, ratio):
    root = os.path.join(data_split_dir, f"forget_{ratio}")
    retain_root = os.path.join(data_split_dir, f"retain_{100 - ratio}")
    return (os.path.join(root, "train-00000-of-00001.parquet"),
            os.path.join(retain_root, "train-00000-of-00001.parquet"))


def build_forget_loader(args, processor):
    forget_parquet, _ = _parquets(args.data_split_dir, args.forget_split_ratio)
    df_forget = pd.read_parquet(forget_parquet)
    idk_list = load_idk()
    print(f"Loaded {len(idk_list)} IDK responses")
    forget_pairs = build_pairs(df_forget)
    forget_ds = PairedDataset(forget_pairs, dpo=True, idk_list=idk_list)
    dl_forget = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=lambda x: collate_dpo(x, processor, args))
    print(f"Forget DPO pairs: {len(forget_ds)}")
    return dl_forget, len(forget_ds)


def build_retain_loader(args, processor, n_samples):
    _, retain_parquet = _parquets(args.data_split_dir, args.forget_split_ratio)
    df_retain = pd.read_parquet(retain_parquet)
    retain_pairs = build_pairs(df_retain)
    retain_ds = PairedDataset(retain_pairs)
    if n_samples == 0:
        raise ValueError("Forget pairs is empty; cannot determine retain sample count.")
    dl_retain = DataLoader(
        retain_ds, batch_size=args.batch_size,
        sampler=RandomSampler(retain_ds, replacement=False, num_samples=n_samples),
        collate_fn=lambda x: collate_plain(x, processor, args),
    )
    print(f"Retain pairs: {len(retain_ds)} (每 epoch 抽 {n_samples})")
    return dl_retain
