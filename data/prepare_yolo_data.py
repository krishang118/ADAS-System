import os
import json
import cv2
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def prepare_bdd100k_for_yolo(bdd_root, output_dir, vehicle_only=True):
    
    bdd_root = Path(bdd_root)
    output_dir = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    if vehicle_only:
        class_map = {'car': 0, 'truck': 1, 'bus': 2, 'train': 3}
        class_names = ['car', 'truck', 'bus', 'train']
    else:
        class_map = {'car': 0, 'truck': 1, 'bus': 2, 'train': 3, 
                     'motorcycle': 4, 'bicycle': 5}
        class_names = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    
    for split in ['train', 'val']:
        possible_json_paths = [
            bdd_root / 'bdd100k_labels_release' / 'bdd100k' / 'labels' / f'bdd100k_labels_images_{split}.json',
            bdd_root / 'labels' / 'det_20' / f'det_{split}.json',
            bdd_root / 'labels' / f'bdd100k_labels_images_{split}.json',
            bdd_root / 'bdd100k' / 'labels' / 'det_20' / f'det_{split}.json',
        ]
        
        json_file = None
        for path in possible_json_paths:
            if path.exists():
                json_file = path
                break
        
        if json_file is None:
            print(f"Warning: No annotation file found for {split}, checked paths:")
            for p in possible_json_paths:
                print(f"  - {p}")
            continue
        
        print(f"Processing {split} set from {json_file}...")
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        
        possible_img_dirs = [
            bdd_root / 'bdd100k' / 'bdd100k' / 'images' / '100k' / split,
            bdd_root / 'images' / '100k' / split,
            bdd_root / 'bdd100k' / 'images' / '100k' / split,
            bdd_root / f'images_{split}',
        ]
        
        img_dir = None
        for path in possible_img_dirs:
            if path.exists():
                img_dir = path
                break
        
        if img_dir is None:
            print(f"Warning: No image directory found for {split}, checked paths:")
            for p in possible_img_dirs:
                print(f"  - {p}")
            continue
        
        converted_count = 0
        for ann in tqdm(annotations, desc=f"Processing {split}"):
            img_name = ann['name']
            img_path = img_dir / img_name
            
            if not img_path.exists():
                continue
            
            shutil.copy(img_path, output_dir / 'images' / split / img_name)
            
            if 'labels' in ann:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                yolo_annotations = []
                for label in ann['labels']:
                    if label['category'] in class_map:
                        class_id = class_map[label['category']]
                        box = label['box2d']
                        
                        x_center = ((box['x1'] + box['x2']) / 2) / w
                        y_center = ((box['y1'] + box['y2']) / 2) / h
                        width = (box['x2'] - box['x1']) / w
                        height = (box['y2'] - box['y1']) / h
                        
                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
                
                if yolo_annotations:
                    label_file = output_dir / 'labels' / split / img_name.replace('.jpg', '.txt')
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    converted_count += 1
        
        print(f"Converted {converted_count} images for {split}")
    
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"Dataset prepared at {output_dir}")
    return str(yaml_path)

def prepare_caltech_for_yolo(caltech_root, output_dir):

    caltech_root = Path(caltech_root)
    output_dir = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    all_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_images.extend(list(caltech_root.rglob(ext)))
    
    all_images = [img for img in all_images if not img.name.startswith('._')]
    
    if len(all_images) == 0:
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['pedestrian']
        }
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        return str(yaml_path)
    
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    for images, split in [(train_images[:1000], 'train'), (val_images[:200], 'val')]:
        for img_path in tqdm(images, desc=f"Processing {split}"):
            try:
                shutil.copy(img_path, output_dir / 'images' / split / img_path.name)
                
                label_file = output_dir / 'labels' / split / (img_path.stem + '.txt')
                label_file.touch()
            except Exception as e:
                continue
    
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['pedestrian']
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"Dataset prepared at {output_dir}")
    return str(yaml_path)

def prepare_mtsd_for_yolo(mtsd_root, output_dir, top_n_classes=13):
   
    mtsd_root = Path(mtsd_root)
    output_dir = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("Processing MTSD dataset...")
    
    sign_classes = [
        'regulatory--stop',
        'regulatory--yield',
        'regulatory--no-entry',
        'warning--general-warning',
        'warning--pedestrian-crossing',
        'regulatory--turn-right',
        'regulatory--turn-left',
        'regulatory--keep-right',
        'regulatory--roundabout',
        'regulatory--speed-limit-30',
        'regulatory--speed-limit-50',
        'regulatory--speed-limit-70',
        'information--parking'
    ]
    
    class_map = {cls: idx for idx, cls in enumerate(sign_classes)}
    
    print("Searching for annotation files...")
    possible_anno_dirs = [
        mtsd_root / 'mtsd_fully_annotated_annotation' / 'mtsd_v2_fully_annotated' / 'annotations',
        mtsd_root / 'annotations',
        mtsd_root / 'mtsd_v2_fully_annotated' / 'annotations'
    ]
    
    anno_dir = None
    for p in possible_anno_dirs:
        if p.exists():
            anno_dir = p
            break
            
    if anno_dir is None:
        print("Annotation directory not found in common paths, searching recursively...")
        json_files = list(mtsd_root.rglob('*.json'))
        annotation_files = [f for f in json_files if len(f.stem) > 10 and 'annotation' not in f.stem and 'stats' not in f.stem]
    else:
        print(f"Found annotation directory: {anno_dir}")
        annotation_files = list(anno_dir.glob('*.json'))
    
    print(f"Total JSON files found: {len(annotation_files)}")
    annotation_files = [f for f in annotation_files if not f.name.startswith('._')]
    print(f"JSON files after filtering ._: {len(annotation_files)}")
    
    if len(annotation_files) > 0:
        print(f"Sample file: {annotation_files[0].name}")
    
    if len(annotation_files) == 0:
        print("Warning: No MTSD annotation files found")
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(sign_classes),
            'names': sign_classes
        }
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        return str(yaml_path)
    
    print(f"Found {len(annotation_files)} annotation files. Parsing and matching images...")
    
    image_map = {}
    print("Indexing images from source directories...")
    image_extensions = ['.jpg', '.jpeg', '.png']
    for ext in image_extensions:
        for img_path in mtsd_root.rglob(f'*{ext}'):
            if not img_path.name.startswith('._'):
                clean_stem = img_path.stem.lstrip('-')
                image_map[clean_stem] = img_path
                
    print(f"Indexed {len(image_map)} images.")
    
    random.shuffle(annotation_files)

    print(f"Processing all {len(annotation_files)} annotation files for full training...")
        
    converted_count = 0
    debug_printed = False
    
    for anno_file in tqdm(annotation_files, desc="Converting MTSD"):
        img_id = anno_file.stem
        
        if img_id not in image_map:
            if not debug_printed:
                print(f"DEBUG: Image ID {img_id} not found in image_map")
                debug_printed = True
            continue
            
        img_path = image_map[img_id]
        
        split = 'train' if random.random() < 0.8 else 'val'
        
        try:
            with open(anno_file, 'r') as f:
                data = json.load(f)
            
            w = data['width']
            h = data['height']
            
            yolo_annotations = []
            
            for obj in data['objects']:
                label = obj['label']
                if not debug_printed and converted_count == 0:
                    print(f"DEBUG: Found label '{label}' in {img_id}")
                
                matched_class_id = None
                for target_lbl, target_id in class_map.items():
                    if label == target_lbl or label.startswith(target_lbl + '--'):
                        matched_class_id = target_id
                        break
                
                if matched_class_id is not None:
                    class_id = matched_class_id
                    bbox = obj['bbox']
                    
                    x_center = ((bbox['xmin'] + bbox['xmax']) / 2) / w
                    y_center = ((bbox['ymin'] + bbox['ymax']) / 2) / h
                    width = (bbox['xmax'] - bbox['xmin']) / w
                    height = (bbox['ymax'] - bbox['ymin']) / h
                    
                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            if yolo_annotations:
                clean_filename = f"{img_id}.jpg"
                
                shutil.copy(img_path, output_dir / 'images' / split / clean_filename)
                
                label_path = output_dir / 'labels' / split / f"{img_id}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                converted_count += 1
                if not debug_printed:
                    print(f"DEBUG: Successfully converted {img_id} to {split}")
                    debug_printed = True
                
        except Exception as e:
            print(f"DEBUG: Error processing {anno_file}: {e}")
            continue
            
    print(f"Successfully converted {converted_count} images with target signs.")

    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(sign_classes),
        'names': sign_classes
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"Dataset prepared at {output_dir}")
    return str(yaml_path)

def prepare_lisa_for_yolo(lisa_root, output_dir):
   
    lisa_root = Path(lisa_root)
    output_dir = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("Processing LISA Traffic Light dataset...")
    
    light_classes = ['red', 'yellow', 'green', 'off']
    class_map = {cls: idx for idx, cls in enumerate(light_classes)}
    
    print("Indexing LISA images...")
    image_map = {}
    image_extensions = ['.jpg', '.jpeg', '.png']
    for ext in image_extensions:
        for img_path in lisa_root.rglob(f'*{ext}'):
            if not img_path.name.startswith('._'):
                image_map[img_path.name] = img_path
    
    print(f"Indexed {len(image_map)} images.")

    anno_files = list(lisa_root.glob('**/*BOX.csv'))
    
    anno_files = [f for f in anno_files if not f.name.startswith('._')]
    
    print(f"Processing all {len(anno_files)} annotation files for full training...")
    
    if not anno_files:
        print("Warning: No LISA annotation files found")
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(light_classes),
            'names': light_classes
        }
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        return str(yaml_path)
    
    print(f"Found {len(anno_files)} annotation files")
    
    import csv
    converted_count = 0
    
    for anno_file in anno_files:
        try:
            with open(anno_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader, None)  
                
                current_img_name = ""
                current_labels = []
                
                rows = list(reader)
                
                img_rows = {}
                for row in rows:
                    if not row: continue
                    filename = Path(row[0]).name 
                    if filename not in img_rows:
                        img_rows[filename] = []
                    img_rows[filename].append(row)
                
                for filename, rows in tqdm(img_rows.items(), desc=f"Processing {anno_file.parent.name}"):
                    
                    if filename not in image_map:
                        continue
                        
                    img_path = image_map[filename]
                    split = 'train' if random.random() < 0.8 else 'val'
                    
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    yolo_annotations = []
                    
                    for row in rows:
                        tag = row[1].lower()
                        x1 = float(row[2])
                        y1 = float(row[3])
                        x2 = float(row[4])
                        y2 = float(row[5])
                        
                        class_id = None
                        if 'stop' in tag: 
                            class_id = 0
                        elif 'warning' in tag: 
                            class_id = 1
                        elif 'go' in tag: 
                            class_id = 2
                        else:
                            continue 
                            
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
                    
                    if yolo_annotations:
                        shutil.copy(img_path, output_dir / 'images' / split / filename)
                        
                        label_path = output_dir / 'labels' / split / (Path(filename).stem + '.txt')
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_annotations))
                        
                        converted_count += 1

        except Exception as e:
            print(f"Error reading {anno_file}: {e}")
            continue
                    
    print(f"Successfully converted {converted_count} images.")

    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(light_classes),
        'names': light_classes
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"Dataset prepared at {output_dir}")
    return str(yaml_path)
