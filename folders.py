import os

splits = ['train', 'test']
classes = ['Drowsy', 'Alert']

for split in splits:
    for cls in classes:
        os.makedirs(f"dataset_binary/{split}/{cls}", exist_ok=True)

print("Folders created successfully.")