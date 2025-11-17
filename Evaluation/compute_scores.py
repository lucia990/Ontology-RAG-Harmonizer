import pandas as pd

irish_local_schema = pd.read_excel('results/irish_local_schema.xlsx')
french_local_schema = pd.read_excel('results/french_local_schema.xlsx')

print(f'Original schema length (IRISH): {len(irish_local_schema)}')
U_irish = set(irish_local_schema['ontology_id'])
U_size_irish = len(U_irish)
print(f'after duplicate_remotion: {U_size_irish}')

print(f'Original schema length (FRENCH): {len(french_local_schema)}')
U_french = set(french_local_schema['ontology_id'])
U_size_french = len(U_french)
print(f'after duplicate_remotion: {U_size_french}')

jaccard_index = len(U_irish.intersection(U_french))/len(U_irish.union(U_french))

print(f'jaccard_index: {jaccard_index}')