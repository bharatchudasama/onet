import pandas as pd
import os

def process_gt_file(path, label_map, output_csv):
    df = pd.read_csv(path)
    df['nevus'] = 1.0 - (df['melanoma'] + df['seborrheic_keratosis'])
    df['label'] = df[['melanoma', 'nevus', 'seborrheic_keratosis']].idxmax(axis=1).map(label_map)
    df_out = df[['image_id', 'label']]
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv} | Total samples: {len(df_out)}")

# Paths to ground truth files (you may need to update these)
train_gt = "datasets/ISIC2017/groundtruth/ISIC-2017_Training_Part3_GroundTruth.csv"
val_gt   = "datasets/ISIC2017/groundtruth/ISIC-2017_Validation_Part3_GroundTruth.csv"
test_gt  = "datasets/ISIC2017/groundtruth/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

output_dir = "datasets/ISIC2017/labels/"
os.makedirs(output_dir, exist_ok=True)

label_map = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratosis': 2}

process_gt_file(train_gt, label_map, os.path.join(output_dir, "train.csv"))
process_gt_file(val_gt, label_map, os.path.join(output_dir, "val.csv"))
process_gt_file(test_gt, label_map, os.path.join(output_dir, "test.csv"))
