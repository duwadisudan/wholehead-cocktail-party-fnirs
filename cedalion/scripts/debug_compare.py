from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data")
sid = "sub-630"
anggyr = BASE / sid / "nirs" / "atlasviewer_mni" / f"anggyr_{sid}.csv"
just = BASE / sid / "nirs" / "atlasviewer_mni" / "just_mni.csv"
print('anggyr ->', anggyr)
print('just ->', just)

df = pd.read_csv(anggyr)
channel_col = 'channel_label' if 'channel_label' in df.columns else df.columns[0]
coords = pd.read_csv(just, header=None).iloc[:,:3].values

print('\nIndex | channel_label | brodmann | scanner | just_mni coord')
for i in range(min(30, len(coords))):
    ch_label = df.iloc[i][channel_col] if i < len(df) else 'MISSING'
    brod = df.iloc[i]['brodmann'] if i < len(df) else 'MISSING'
    scan = df.iloc[i]['scanner'] if i < len(df) and 'scanner' in df.columns else None
    print(f"{i:2d} | {ch_label:12} | {str(brod):30} | {str(scan):25} | {coords[i]}")

print('\nSummary:')
print('Number of anggyr rows in anggyr file:', len(df[df['brodmann'].apply(lambda x: (str(x).split('-')[-1].split('(')[0].strip()=='AngGyrus'))]))
print('just_mni rows:', coords.shape[0])
