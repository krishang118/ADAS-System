import os
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

class UTARLDDDataset:
    
    def __init__(self, root_dir: Path = None):
      
        if root_dir is None:
            root_dir = Config.UTA_RLDD_ROOT
        
        self.root_dir = Path(root_dir)
        self.fold1_part1 = self.root_dir / "Fold1_part1"
        self.fold1_part2 = self.root_dir / "Fold1_part2"
        
        self.samples = self._load_samples()
        
        print(f"Loaded UTA-RLDD dataset from {self.root_dir}")
        print(f"   Total videos: {len(self.samples)}")
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        
        for part_dir in [self.fold1_part1, self.fold1_part2]:
            if not part_dir.exists():
                print(f"Warning: {part_dir} not found")
                continue
            
            for subject_dir in sorted(part_dir.iterdir()):
                if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                    continue
                
                subject_id = subject_dir.name
                
                video_files = list(subject_dir.glob('*.[mM][oO][vV]')) + list(subject_dir.glob('*.[mM][pP]4'))
                
                for video_file in video_files:
                    if video_file.name.startswith('._'):
                        continue
                    
                    filename = video_file.stem
                    
                    if filename == '0':
                        state = 'Alert'
                    elif filename == '5':
                        state = 'Fatigued'
                    elif filename == '10':
                        state = 'Drowsy'
                    else:
                        state = 'Unknown'
                    
                    samples.append({
                        'video_path': str(video_file),
                        'filename': video_file.name,
                        'state': state,
                        'subject': subject_id,
                        'part': 'part1' if 'part1' in str(part_dir) else 'part2'
                    })
        
        return samples
    
    def get_samples_by_state(self, state: str) -> List[Dict]:
        return [s for s in self.samples if s['state'] == state]
    
    def get_samples_by_subject(self, subject: str) -> List[Dict]:
        return [s for s in self.samples if s['subject'] == subject]
    
    def get_video_frames(self, video_path: str, max_frames: int = None, target_fps: int = 10):
    
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            original_fps = 30  
        
        frame_skip = max(1, int(original_fps / target_fps))
        
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame = cv2.resize(frame, (Config.DSM_FRAME_WIDTH, Config.DSM_FRAME_HEIGHT))
                yield frame
                processed_count += 1
                
                if max_frames and processed_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
    
    def get_state_distribution(self) -> Dict[str, int]:
        distribution = {}
        for sample in self.samples:
            state = sample['state']
            distribution[state] = distribution.get(state, 0) + 1
        return distribution
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
