"""
Préparation des données de validation 2D pour BraTS2018
Modification pour Ibrahim (differentes dataset HGG/LGG): 
Author: Ibrahim
"""


import os
import numpy as np
import SimpleITK as sitk

flair_name = "_flair.nii.gz"
t1_name    = "_t1.nii.gz"
t1ce_name  = "_t1ce.nii.gz"
t2_name    = "_t2.nii.gz"
mask_name  = "_seg.nii.gz"

h, w = 224, 224

val_root = r"./data/MICCAI_BraTS_2018_Data_Validation"  # <-- le dossier principale contenant les sous-dossiers des patients
outputImg_path  = r"./data/BraTS2018_split/val/image"
outputMask_path = r"./data/vBraTS2018_split/val/mask"

os.makedirs(outputImg_path, exist_ok=True)
os.makedirs(outputMask_path, exist_ok=True)

def normalize(vol, bottom=99, down=1):
    b = np.percentile(vol, bottom)
    t = np.percentile(vol, down)
    vol = np.clip(vol, t, b)

    vol_nonzero = vol[np.nonzero(vol)]
    if np.std(vol) == 0 or vol_nonzero.size == 0 or np.std(vol_nonzero) == 0:
        return vol
    tmp = (vol - np.mean(vol_nonzero)) / np.std(vol_nonzero)
    tmp[tmp == tmp.min()] = -9
    return tmp

def crop_center(vol, croph, cropw):
    # vol: (Z,H,W)
    _, H, W = vol.shape
    starth = H // 2 - croph // 2
    startw = W // 2 - cropw // 2
    return vol[:, starth:starth + croph, startw:startw + cropw]

def list_cases(root):
    # garde uniquement les sous-dossiers (ignore .DS_Store etc.)
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")]

cases = list_cases(val_root)

for case_id in cases:
    case_dir = os.path.join(val_root, case_id)

    flair_path = os.path.join(case_dir, case_id + flair_name)
    t1_path    = os.path.join(case_dir, case_id + t1_name)
    t1ce_path  = os.path.join(case_dir, case_id + t1ce_name)
    t2_path    = os.path.join(case_dir, case_id + t2_name)
    seg_path   = os.path.join(case_dir, case_id + mask_name)

    # Vérifie que les 4 modalités existent
    needed = [flair_path, t1_path, t1ce_path, t2_path]
    if not all(os.path.exists(p) for p in needed):
        print(f"[SKIP] {case_id}: fichiers manquants")
        continue

    # Charge volumes
    flair = sitk.GetArrayFromImage(sitk.ReadImage(flair_path, sitk.sitkInt16))
    t1    = sitk.GetArrayFromImage(sitk.ReadImage(t1_path, sitk.sitkInt16))
    t1ce  = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_path, sitk.sitkInt16))
    t2    = sitk.GetArrayFromImage(sitk.ReadImage(t2_path, sitk.sitkInt16))

    # Normalisation par modalité
    flair = normalize(flair)
    t1    = normalize(t1)
    t1ce  = normalize(t1ce)
    t2    = normalize(t2)

    # Crop centre
    flair = crop_center(flair, h, w)
    t1    = crop_center(t1, h, w)
    t1ce  = crop_center(t1ce, h, w)
    t2    = crop_center(t2, h, w)

    has_seg = os.path.exists(seg_path)
    if has_seg:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path, sitk.sitkUInt8))
        seg = crop_center(seg, h, w)

    print(f"[OK] {case_id} | seg={'yes' if has_seg else 'no'} | slices={flair.shape[0]}")

    for z in range(flair.shape[0]):
        # Si seg existe: on garde seulement slices avec tumeur (comme ton code)
        if has_seg:
            if np.max(seg[z]) == 0:
                continue
            mask_slice = seg[z].astype(np.uint8)

        # Construit (H,W,4)
        img4 = np.zeros((h, w, 4), dtype=np.float32)
        img4[:, :, 0] = flair[z].astype(np.float32)
        img4[:, :, 1] = t1[z].astype(np.float32)
        img4[:, :, 2] = t1ce[z].astype(np.float32)
        img4[:, :, 3] = t2[z].astype(np.float32)

        np.save(os.path.join(outputImg_path, f"{case_id}_{z}.npy"), img4)

        # Sauve masque seulement si GT dispo
        if has_seg:
            np.save(os.path.join(outputMask_path, f"{case_id}_{z}.npy"), mask_slice)

print("Done.")
