import pandas as pd
import os

ppi_file    = '9606.protein.links.v12.0.txt'
mapping_file= 'string_mapping.tsv'
out_file    = 'human_ppi_mapped_deduplicated.tsv'

ppi = pd.read_csv(ppi_file, sep=' ', header=None, names=['protein1','protein2','combined_score_raw'], dtype=str, low_memory=False)
ppi['combined_score_raw'] = pd.to_numeric(ppi['combined_score_raw'], errors='coerce').dropna()

mapping = pd.read_csv(mapping_file, sep='\t', header=0, usecols=[2,3], names=['stringId','preferredName']).dropna()
id2name = dict(zip(mapping.stringId, mapping.preferredName))

ppi['Gene1'] = ppi.protein1.map(id2name)
ppi['Gene2'] = ppi.protein2.map(id2name)
ppi = ppi.dropna(subset=['Gene1','Gene2'])

ppi['combine_score'] = ppi.combined_score_raw / 1000
ppi['ordered_pair'] = ppi.apply(lambda r: tuple(sorted([r.Gene1, r.Gene2])), axis=1)
ppi = ppi.drop_duplicates('ordered_pair')

ppi[['Gene1','Gene2','combine_score']].to_csv(out_file, sep='\t', index=False)
print(f'Done. Deduplicated PPI saved to {os.path.abspath(out_file)}')