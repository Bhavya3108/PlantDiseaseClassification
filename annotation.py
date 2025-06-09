import os
import pandas as pd

# Paths
base_path = 'G:/PlantDisease'
data_dir = os.path.join(base_path, 'PlantifyDr')
output_csv_path = os.path.join(base_path, 'plantifydr_annotations.csv')

# Detect classes
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(classes)
class_id_map = {name: idx for idx, name in enumerate(classes)}
print(f"Number of classes: {num_classes}")
print("Class ID mapping:")
for class_name, class_id in class_id_map.items():
    print(f"Class {class_name}: ID {class_id}")

# Collect data
data = []
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img in images:
        data.append({
            "image_filename": img,
            "label": class_name,
            "class_id": class_id_map[class_name]
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv_path, index=False)

# Verify the output
print(f"\nAnnotation file saved to: {output_csv_path}")
print(f"Total entries: {len(df)}")
print("\nSample of the CSV file:")
print(df.head())