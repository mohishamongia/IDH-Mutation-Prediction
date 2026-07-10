import pandas as pd
import os

def load_ucsf_pdgm_metadata(csv_path, data_dir):
    df = pd.read_csv("DATASETS/UCSF-PDGM-v5/UCSF-PDGM-metadata_v5.csv")  

 
    df = df[df["IDH"].notna()]
    df = df[df["IDH"] != "unknown"]

   
    df["idh_label"] = df["IDH"].apply(
        lambda x: 0 if str(x).strip().lower() == "wildtype" else 1
    )

    
    df = df[df["ID"].apply(
        lambda s: os.path.isdir(os.path.join(data_dir, s)))]

    out = pd.DataFrame({
        "subject_id": df["ID"],
        "dataset": "UCSF-PDGM",
        "idh_label": df["idh_label"],
        "age": df["Age at MRI"],
        "sex": df["Sex"],
        "who_grade": df["WHO CNS Grade"],
        "codeletion_1p19q": df["1p/19q"],
    })
    return out

ucsf_meta = load_ucsf_pdgm_metadata(
    csv_path="DATASETS/UCSF-PDGM-v5/UCSF-PDGM-metadata_v5.csv",
    data_dir="DATASETS/UCSF-PDGM-v5"
)

print(f"Total: {len(ucsf_meta)}")
print(ucsf_meta["idh_label"].value_counts())
print(f"Mutated: {ucsf_meta['idh_label'].sum()} "
      f"({100*ucsf_meta['idh_label'].mean():.1f}%)")