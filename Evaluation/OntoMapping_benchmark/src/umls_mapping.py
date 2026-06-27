import pandas as pd
from collections import Counter
import numpy as np
import argparse
import ast
import os

''' Count most common mappings within the UMLS
umls_df = pd.read_csv('UMLS_mapper/data/raw/filtered_conso_eng.csv', index_col = False, names= ['CUI', 'Language', 'Status', 'LUI', 'string_type', 'SUI', 'atom_status', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'source', 'type', 'CODE', 'Name', 'restriction_level', 'SUPPRESS', 'CVF'], low_memory=False)
umls_df = umls_df[['Name', 'CUI', 'source', 'SCUI', 'type']]

N = 30
grouping = umls_df.groupby('CUI')['source'].apply(list)
grouping.apply(set)
list_voc_group = grouping.apply(set).to_list()
counts = Counter(tuple(s) for s in list_voc_group)
common_voc_tuples = counts.most_common(N)
'''


def create_mapping_table(vocabularies: str, concept_filename: str = 'filtered_conso_eng.csv'):
    print(f'Creating mapping table for ontologies {vocabularies}...')
    list_vocabularies = vocabularies.split()
    print('Loading UMLS concepts...')
    umls_df = pd.read_csv(f"UMLS_mapper/data/raw/{concept_filename}", index_col=False,
                       names=['CUI', 'Language', 'Status', 'LUI', 'string_type', 'SUI', 'atom_status', 'AUI', 'SAUI',
                              'SCUI', 'SDUI', 'source', 'type', 'CODE', 'Name', 'restriction_level', 'SUPPRESS', 'CVF'], low_memory=False)
    umls_df = umls_df[['Name', 'CUI', 'source', 'CODE', 'type']]
    print('UMLS vocabularies loaded...')
    # Assess vocabularies exist in the UMLS

    all_onto = np.array(umls_df.source.unique())
    if np.isin(list_vocabularies, all_onto).all():
        # Filter dataframe per vocabulary
        filtered_umls = umls_df[umls_df.source.isin(list_vocabularies)].reset_index(drop=True)

        # Count codes per CUI to filter just 1:1 mappings
        filtered_umls = filtered_umls[['CUI', 'Name', 'source', 'CODE']].drop_duplicates()

        filtered_umls['source'] = pd.Categorical(
            filtered_umls['source'],
            categories=list_vocabularies,
            ordered=True
        )
        mask = filtered_umls.groupby('CUI')['source'].transform('nunique') > 1
        # extract synonyms from UMLS
        grouping = (
            filtered_umls[mask]
            .sort_values(['CUI', 'source'])
            .groupby('CUI', sort=False)
            .agg(
                sources=('source', list),
                ids=('CODE', list),
                names=('Name', list)
            )
        )
        mapping = (
            grouping
            .explode(['sources', 'ids', 'names'])
            .pivot_table(index='CUI', columns='sources', values=['ids','names'] , aggfunc=list)
        )
        mapping.columns = [f"{vocab}_{val}" for val, vocab in mapping.columns]
        mapping.columns.name = None
        if len(mapping) > 0:
            print(f'Mapping table dimensions: {mapping.shape}')
            mapping.to_csv(f'Evaluation/OntoMapping_benchmark/mapped_codes/mapping_{"_".join(list_vocabularies)}.csv')
            print(f"Mapping saved to Evaluation/OntoMapping_benchmark/mapped_codes/mapping_{'_'.join(list_vocabularies)}.csv")
            return mapping
        else:
            print('No mapping available for the provided vocabularies')
            return 0
    else:
        print('No valid vocabularies provided!')
        return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ontologies', type=str, required=True)
    parser.add_argument('--concept_filename', type=str, default='filtered_conso_eng.csv')
    args = parser.parse_args()
    list_vocabularies = args.ontologies
    concept_filename = args.concept_filename

    create_mapping_table(list_vocabularies, concept_filename=concept_filename)