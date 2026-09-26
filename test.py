import json
import pandas as pd

df = pd.read_parquet("data_skincare/eval_sample.parquet")
pick = df.groupby("segment").head(5)                 # Face 5개, Body 5개 = 10개
targets = dict(zip(pick["parent_asin"], pick["title"]))

found = {}
with open("meta_Beauty_and_Personal_Care.jsonl", encoding="utf-8") as f:
    for line in f:
        m = json.loads(line)
        if m["parent_asin"] in targets:
            imgs = m["images"]
            if isinstance(imgs, dict):               # hi_res까지 포함해서 변환
                imgs = [dict(zip(imgs.keys(), v)) for v in zip(*imgs.values())]
            found[m["parent_asin"]] = [
                (im.get("variant"), im.get("hi_res") or f"(hi_res 없음) {im.get('large')}")
                for im in imgs
            ]
            if len(found) == len(targets):
                break

for asin, imgs in found.items():
    print(f"\n[{asin}] {targets[asin][:70]}")
    for v, u in imgs:
        print(f"  {v:6s} {u}")