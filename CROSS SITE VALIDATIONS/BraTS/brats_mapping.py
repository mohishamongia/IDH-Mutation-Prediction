'''import pandas as pd
import os

# Load mapping file
mapping = pd.read_excel('BraTS2023_2017_GLI_Mapping.xlsx')

# Get validation subject IDs
val_dir = '/workspace/BraTS2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/'
val_subjects = os.listdir(val_dir)
print(f'Total validation subjects: {len(val_subjects)}')

# Cross reference with mapping
matched = mapping[mapping['BraTS2023'].isin(val_subjects)]
print(f'Matched in mapping file: {len(matched)}')

# Check cohort distribution of matched
print('\nCohort distribution:')
print(matched['Cohort Name (if publicly available)'].value_counts())

# Check how many have BraTS2021 IDs
has_2021 = matched[matched['BraTS2021'].notna()]
print(f'\nHave BraTS2021 ID: {len(has_2021)}')
print(has_2021[['BraTS2023', 'BraTS2021', 'Cohort Name (if publicly available)']].head(10))'''


'''import requests

r = requests.get(
    "https://www.cbioportal.org/api/studies/lgg_tcga/clinical-data",
    params={"clinicalAttributeId": "IDH1_MUTATION_FOUND", "type": "PATIENT"},
    headers={"Accept": "application/json"},
    timeout=30
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Records: {len(data)}")
    print("Sample:", data[:3])
else:
    print(r.text[:200])'''


import requests
import pandas as pd

# Get all available clinical attributes for LGG
r = requests.get(
    "https://www.cbioportal.org/api/studies/lgg_tcga/clinical-attributes",
    headers={"Accept": "application/json"},
    timeout=30
)
attrs = r.json()

# Find IDH related ones
idh_attrs = [a for a in attrs if 'IDH' in a.get('clinicalAttributeId', '').upper() 
             or 'IDH' in a.get('displayName', '').upper()]
print("IDH attributes in LGG:")
for a in idh_attrs:
    print(f"  {a['clinicalAttributeId']}: {a['displayName']}")

print()

# Same for GBM
r2 = requests.get(
    "https://www.cbioportal.org/api/studies/gbm_tcga/clinical-attributes",
    headers={"Accept": "application/json"},
    timeout=30
)
attrs2 = r2.json()
idh_attrs2 = [a for a in attrs2 if 'IDH' in a.get('clinicalAttributeId', '').upper()
              or 'IDH' in a.get('displayName', '').upper()]
print("IDH attributes in GBM:")
for a in idh_attrs2:
    print(f"  {a['clinicalAttributeId']}: {a['displayName']}")