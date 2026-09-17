"""Design train/test split for COX-2 dataset with scaffold diversity."""
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

with open("data/processed/metadata.json") as f:
    meta = json.load(f)
data = np.load("data/processed/features.npz", allow_pickle=True)
y = data["y"]
sids = np.array(meta["scaffold_ids"])
unique_scaffolds = meta["unique_scaffolds"]
smiles = meta["smiles"]

scaffold_info = {}
for idx in range(len(y)):
    sid = sids[idx]
    if sid not in scaffold_info:
        scaffold_info[sid] = {"pos": [], "neg": [], "smi": unique_scaffolds[sid]}
    if y[idx] == "Pos":
        scaffold_info[sid]["pos"].append(idx)
    else:
        scaffold_info[sid]["neg"].append(idx)

train_sids = {40, 29, 27, 84}

train_scaffold_fps = []
for sid in train_sids:
    mol = Chem.MolFromSmiles(unique_scaffolds[sid])
    if mol:
        train_scaffold_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))

candidates = []
for sid, info in scaffold_info.items():
    if sid in train_sids:
        continue
    n = len(info["pos"]) + len(info["neg"])
    if n < 1:
        continue
    mol = Chem.MolFromSmiles(info["smi"])
    if not mol:
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    max_sim = max(DataStructs.TanimotoSimilarity(fp, tfp) for tfp in train_scaffold_fps)
    candidates.append({
        "sid": sid, "n": n, "pos": len(info["pos"]), "neg": len(info["neg"]),
        "max_sim": max_sim, "scaffold": info["smi"][:60],
    })

candidates.sort(key=lambda x: x["max_sim"])

print("Top 30 most different scaffolds from training:")
print(f"{'SID':<5} {'N':>3} {'P':>3} {'Ng':>3} {'MaxSim':>7} Scaffold")
for c in candidates[:30]:
    print(f"{c['sid']:<5} {c['n']:>3} {c['pos']:>3} {c['neg']:>3} {c['max_sim']:>7.3f} {c['scaffold']}")

rng = np.random.RandomState(42)
test_pos, test_neg = [], []
used_sids = set()

for c in candidates:
    if c["sid"] in used_sids:
        continue
    if len(test_pos) < 5 and c["pos"] > 0 and c["max_sim"] < 0.6:
        idx = rng.choice(scaffold_info[c["sid"]]["pos"])
        test_pos.append((idx, c["sid"], c["max_sim"]))
        used_sids.add(c["sid"])
    elif len(test_neg) < 5 and c["neg"] > 0 and c["max_sim"] < 0.6:
        idx = rng.choice(scaffold_info[c["sid"]]["neg"])
        test_neg.append((idx, c["sid"], c["max_sim"]))
        used_sids.add(c["sid"])
    if len(test_pos) >= 5 and len(test_neg) >= 5:
        break

print(f"\nSelected test Pos ({len(test_pos)}):")
for idx, sid, sim in test_pos:
    print(f"  idx={idx} sid={sid} sim={sim:.3f} label={y[idx]} smi={smiles[idx][:60]}")
print(f"Selected test Neg ({len(test_neg)}):")
for idx, sid, sim in test_neg:
    print(f"  idx={idx} sid={sid} sim={sim:.3f} label={y[idx]} smi={smiles[idx][:60]}")

test_idx = sorted([i for i, _, _ in test_pos] + [i for i, _, _ in test_neg])
test_sids = set(s for _, s, _ in test_pos + test_neg)

# Training: ~100 from train_sids, imbalanced
test_set = set(test_idx)
train_pool_pos, train_pool_neg = [], []
for sid in train_sids:
    for i in scaffold_info[sid]["pos"]:
        if i not in test_set:
            train_pool_pos.append(i)
    for i in scaffold_info[sid]["neg"]:
        if i not in test_set:
            train_pool_neg.append(i)

keep_pos = rng.choice(train_pool_pos, min(8, len(train_pool_pos)), replace=False).tolist()
keep_neg = rng.choice(train_pool_neg, min(92, len(train_pool_neg)), replace=False).tolist()
train_idx = sorted(keep_pos + keep_neg)

tp = sum(1 for i in train_idx if y[i] == "Pos")
tn = sum(1 for i in train_idx if y[i] == "Neg")
print(f"\nTrain: {len(train_idx)} (Pos={tp}, Neg={tn}, ratio={tn/max(tp,1):.1f}:1)")
print(f"Test:  {len(test_idx)} (Pos={len(test_pos)}, Neg={len(test_neg)})")
print(f"Test scaffolds: {test_sids}")
print(f"Train scaffolds: {train_sids}")
print(f"Overlap: {test_sids & train_sids}")

test_str = ",".join(str(i) for i in test_idx)
train_str = ",".join(str(i) for i in train_idx)
print(f"\n--test-indices \"{test_str}\"")
print(f"--train-indices \"{train_str}\"")
