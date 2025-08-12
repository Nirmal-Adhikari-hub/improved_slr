from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import video_aug


class PhoenixFeeder(Dataset):
    """
    Functional equivalent to slowfast's BaseFeeder, with clean DI and no
    global state.
    """

    def __init__(self,
                 dataset_root: str,
                 preprocess_root: str,
                 gloss_dict_path: str,
                 dataset: str = "phoenix2014",
                 mode: str = "train",
                 datatype: str = "video",           # "video" | "feaures" | "lmdb (not implemented)"
                 frame_interval: int = 1,
                 image_scale: float = 1.0,
                 input_size: int = 224,
                 kernel_spec: Optional[List[str]] = None, # eg. ["K5","P2","K5","P2"]
                 transform_train: bool = True,
                 frame_subdir: Optional[str] = "features/fullFrame-256x256px"   
    ):
        self.dataset = dataset
        self.mode = mode
        self.datatype = datatype
        self.frame_interval = frame_interval
        self.image_scale = image_scale
        self.input_size = input_size
        self.kernel_spec = kernel_spec or []
        self.transform_mode = "train" if transform_train else "test"

        # Paths
        self.dataset_root = Path(dataset_root)
        self.preprocess_root = Path(preprocess_root)
        self.frame_root = self.dataset_root / (frame_subdir if frame_subdir else "")

        # metadata and vocab
        self.inputs_list: Dict[Any, Any] = np.load(
            str(self.preprocess_root / dataset / f"{mode}_info.npy"),
            allow_pickle=True,
        ).item()
        self.gloss_dict: Dict[str, List[int]] = np.load(gloss_dict_path,
                                                        allow_pickle=True).item()
        
        # Augs
        self.data_aug = self._build_transform()
        print(
            f"[PhoenixFeeder] {mode}: {len(self)} samples | interval={frame_interval} | "
            f"scale={image_scale} | input={input_size}"
        )

    def _build_transform(self):
        if self.transform_mode == "train":
            print("Applying training transformations")
            return video_aug.Compose(
                [
                    video_aug.RandomCrop(self.input_size),
                    video_aug.RandomHorizontalFlip(0.5),
                    video_aug.Resize(self.image_scale),
                    video_aug.ToTensor(),
                    video_aug.TemporalRescale(0.2, self.frame_interval)
                ]
            )
        else:
            print("Applying test transformations")
            return video_aug.Compose(
                [
                    video_aug.CenterCrop(self.input_size),
                    video_aug.Resize(self.image_scale),
                    video_aug.ToTensor(),
                ]
            )
        
    def __len__(self):
        return len(self.inputs_list) - 1
    
    # -------------------- IO --------------------

    def _label_to_ids(self, label_str:str) -> List[int]:
        ids = []
        for tok in label_str.split(" "):
            if tok and tok in self.gloss_dict:
                ids.append(self.gloss_dict[tok][0])
        return ids
    
    def _read_video(self, index: int):
        fi = self.inputs_list[index]
        # glob rule
        if "phoenix" in self.dataset.lower():
            # join our chosen frame root (user can switch 210x260 vs 256x256) 
            # with fi["folder"] (contains 'train/.../*.png', or '*.jpg')
            img_folder = os.path.join(str(self.frame_root), fi["folder"])
        elif self.dataset == "CSL":
            img_folder = os.path.join(str(self.frame_root), fi["folder"], "*.jpg")
        elif self.dataset == "CSL-Daily":
            img_folder = os.path.join(str(self.dataset_root), fi["folder"])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        img_list = sorted(glob.glob(img_folder))

        # random offset + frame_interval sampling
        if self.frame_interval > 1:
            offset = int(torch.randint(0, self.frame_interval, [1]))
            img_list = img_list[offset::self.frame_interval]
        # labels
        label_ids = self._label_to_ids(fi["label"])
        # RGB frames
        if self.dataset != 'CSL-Daily':
            frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                      for p in img_list]
        else:
            frames = [
                cv2.cvtColor(cv2.resize(cv2.imread(p)[40:, ...], (256,256)), cv2.COLOR_BGR2RGB)
                for p in img_list            
            ]
        return frames, label_ids, fi
    
    def _read_features(self, index: int):
        fi = self.inputs_list[index]
        # ./features/{mode}/{fileid}_features.npy (relative to project root)
        data = np.load(
            str(self.preprocess_root.parent / "features" / self.mode / f"{fi['fileid']}_features.npy"),
            allow_pickle=True,
        ).item()
        return data["features"], data["label"]
    
    def _normalize_(self, video_frames, label_ids, file_id: Optional[str] = None):
        # Apply same compose; only WERAugment uses label/file_id (we left it available)
        video_t, label_ids = self.data_aug(video_frames, label_ids, file_id)
        # ((x/255) - 0.45) / 0.225
        video_t = ((video_t.float() / 255.) - 0.45) / 0.225
        return video_t, label_ids
    
    def __getitem__(self, idx: int):
        if self.datatype =="video":
            frames, label_ids, fi = self._read_video(idx)
            video_t, label_ids = self._normalize_(frames, label_ids, fi.get("fileid"))
            return video_t, torch.LongTensor(label_ids), self.inputs_list[idx]["original_info"]
        elif self.datatype == "features":
            feats, label_ids = self._read_features(idx)
            return feats, torch.LongTensor(label_ids), self.inputs_list[idx]["original_info"]
        else:
            raise NotImplementedError("datatype='lmdb' path is not implemented in this port.")
        

# -------------------- Collate ---------------------


def _compute_padding_and_stride(kernel_spec: List[str]):
    left_pad = 0
    last_stride = 1
    total_stride = 1
    for ks in (kernel_spec or []):
        if ks[0] == 'K':
            left_pad = left_pad * last_stride
            left_pad += int((int(ks[1:]) - 1) / 2)
        elif ks[0] == 'P':
            last_stride = int(ks[1:])
            total_stride *= last_stride
    return left_pad, total_stride


def make_collate_fn(kernel_spec: List[str], is_video: bool = True):
    left_pad, total_stride = _compute_padding_and_stride(kernel_spec or [])

    def collate(batch):
        # sort by length desc
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        video, label, info = list(zip(*batch))

        if is_video and len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [int(np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad) for vid in video]
            )
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len_pad = max_len + left_pad + right_pad
            padded_video = [
                torch.cat(
                    (
                        vid[0][None].expand(left_pad, -1, -1, -1),
                        vid,
                        vid[-1][None].expand(max_len_pad - len(vid) - left_pad, -1, -1, -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video, dim=0)
        else:
            # feature tensors (T,D) -> pad to max_T and permute to (B,D,T)
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [
                torch.cat((vid, vid[-1][None].expand(max_len - len(vid), -1)), dim=0) for vid in video
            ]
            padded_video = torch.stack(padded_video, dim=0).permute(0, 2, 1)

        label_length = torch.LongTensor([len(lab) for lab in label])
        if int(label_length.max()) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    return collate