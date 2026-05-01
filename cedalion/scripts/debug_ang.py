from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data")
sid = "sub-630"
anggyr = BASE / sid / "nirs" / "atlasviewer_mni" / f"anggyr_{sid}.csv"
just = BASE / sid / "nirs" / "atlasviewer_mni" / "just_mni.csv"
print('anggyr path ->', anggyr)
print('just path ->', just)

df = pd.read_csv(anggyr)
print("Columns:", df.columns.tolist())
channel_col = 'channel_label' if 'channel_label' in df.columns else df.columns[0]

def extract_ba_name(label):
    if pd.isna(label) or label=="Outside defined BAs":
        return label if pd.notna(label) else "Unknown"
    parts = str(label).split("-")
    if len(parts)>1:
        return parts[1].split("(")[0].strip()
    return str(label).split("(")[0].strip()

ang = df[df['brodmann'].apply(lambda x: extract_ba_name(x)=="AngGyrus")]
print(f"Found {len(ang)} AngGyrus rows (should be ~58).")
indices=[]
for idx, r in ang.iterrows():
    lab=r[channel_col]
    try:
        ch_idx=int(lab)
    except:
        try:
            ch_idx=int(idx)
        except:
            ch_idx=None
    indices.append((idx, lab, ch_idx, r['brodmann'], r.get('scanner', None)))

print("Sample rows (idx, label, numeric_index, brodmann, scanner):")
for i in indices[:12]:
    print(i)

coords = pd.read_csv(just, header=None).iloc[:,:3].values
valid_idx = [ci for (_,_,ci,_,_) in indices if ci is not None and 0<=ci<coords.shape[0]]
print("Numeric indices used:", valid_idx[:20], "count:", len(valid_idx))
if valid_idx:
    pts = coords[valid_idx,:]
    centroid = pts.mean(axis=0)
    print("Centroid (MNI):", centroid)
    dists = np.linalg.norm(pts-centroid[None,:],axis=1)
    print("mean radial distance:", dists.mean(), "min/max:", dists.min(), dists.max())
    print("First 10 per-channel coords:")
    for p in pts[:10]:
        print(p)
else:
    print("No numeric indices available for centroid.")
