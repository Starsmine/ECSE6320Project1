import pandas as pd
import re

BYTE_WIDTH={"float32":4,"float64":8,"int32":4}
CSV='saxpy_benchmark.csv'

df=pd.read_csv(CSV)
df.columns=[c.strip() for c in df.columns]

# normalize pattern (separate gather variants)
def normalize_pattern(v, separate=True):
    if pd.isna(v):
        return ''
    s=str(v).strip()
    s=re.sub(r'^["\']|["\']$', '', s).lower().strip()
    if re.search(r'contig',s):
        return 'contiguous'
    if re.search(r'stride',s):
        return 'stride'
    if re.search(r'gather',s):
        return s if separate else 'gather'
    return s

if 'Pattern' in df.columns:
    df['Pattern']=df['Pattern'].map(lambda v: normalize_pattern(v, True))
else:
    df['Pattern']=''

# touched computation
MAX_STRIDE_FOR_GATHER=8

def compute_touched(row):
    pattern=str(row.get('Pattern','')).lower()
    try:
        stride=int(row.get('Stride',0))
    except Exception:
        stride=0
    n=int(row.get('N',0))
    if 'stride' in pattern:
        if stride<=0:
            return max(1,n)
        return max(1, n//max(1,stride))
    if 'gather' in pattern:
        if n<=1:
            return 1
        return max(1,(n-1)//MAX_STRIDE_FOR_GATHER)
    return max(1,n)

if 'Stride' not in df.columns:
    df['Stride']=0

df['Touched']=df.apply(compute_touched, axis=1)
df['Size_KB']=df['Touched']*df['Type'].map(BYTE_WIDTH)/1024.0

# group and summary as script does

group_cols=["Kernel","Type","N","Aligned","Size_KB","Pattern"]
agg_funcs={"GFLOP/s":["mean","std","median","count"]}
summary=df.groupby(group_cols).agg(agg_funcs)
summary.columns=["_".join(c).strip() for c in summary.columns.values]
summary=summary.reset_index()

# classify kernel
def classify_kernel(name, separate_gather_variants=False):
    ln=(name or '').lower()
    if 'gather' in ln and not separate_gather_variants:
        return 'gather'
    if 'stride' in ln:
        return 'stride'
    return 'base'

summary['Group']=summary['Kernel'].map(lambda k: classify_kernel(k, True))

print('Groups and counts:')
print(summary['Group'].value_counts())
print('\nExample Kernel names where Group==stride (first 20):')
print(summary[summary['Group']=='stride']['Kernel'].unique()[:20])
print('\nExample Patterns present (unique):')
print(sorted(summary['Pattern'].unique())[:40])
