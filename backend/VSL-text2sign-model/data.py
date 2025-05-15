import sys
import os
import io
import re
from typing import Optional
import glob
import numpy as np
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, TARGET_PAD

from vocabulary import Vocabulary, build_vocab

# Thiết lập logger
logger = logging.getLogger(__name__)

def load_data(cfg: dict) -> tuple[Dataset, Dataset, Optional[Dataset], Vocabulary, list]:
    """
    Load train, dev và optionally test data như được chỉ định trong configuration.
    Vocabularies được tạo từ tập huấn luyện với giới hạn `voc_limit` tokens và tần suất tối thiểu `voc_min_freq`.

    :param cfg: từ điển cấu hình cho dữ liệu (phần "data" của file cấu hình)
    :return:
        - train_data: tập dữ liệu huấn luyện
        - dev_data: tập dữ liệu phát triển
        - test_data: tập dữ liệu kiểm tra nếu có, nếu không thì None
        - src_vocab: từ vựng nguồn được trích xuất từ dữ liệu huấn luyện
        - trg_vocab: từ vựng đích (ở đây là danh sách các None có kích thước trg_size)
    """
    data_cfg = cfg["data"]
    max_sent_length = data_cfg["max_sent_length"]
    trg_size = cfg["model"]["trg_size"] + 1  # +1 cho counter
    skip_frames = data_cfg.get("skip_frames", 1)

    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    print(f"Loading data from: train={train_path}, dev={dev_path}, test={test_path}")

    # Tạo các dataset
    train_dataset = SignProdDataset(
        path=train_path,
        trg_size=trg_size,
        skip_frames=skip_frames,
        max_sent_length=max_sent_length
    )
    print(f"Loaded {len(train_dataset)} training samples")

    dev_dataset = SignProdDataset(
        path=dev_path,
        trg_size=trg_size,
        skip_frames=skip_frames,
        max_sent_length=max_sent_length
    )
    print(f"Loaded {len(dev_dataset)} dev samples")

    test_dataset = SignProdDataset(
        path=test_path,
        trg_size=trg_size,
        skip_frames=skip_frames,
        max_sent_length=max_sent_length
    )
    print(f"Loaded {len(test_dataset)} test samples")

    # Xây dựng từ vựng nguồn
    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(
        field="src",
        max_size=src_max_size,
        min_freq=src_min_freq,
        dataset=train_dataset,
        vocab_file=src_vocab_file
    )

    # Từ vựng đích là danh sách các None với kích thước trg_size
    trg_vocab = [None] * trg_size

    train_dataset.src_vocab = src_vocab
    train_dataset.trg_size = trg_size
    dev_dataset.src_vocab = src_vocab
    dev_dataset.trg_size = trg_size
    test_dataset.src_vocab = src_vocab
    test_dataset.trg_size = trg_size

    return train_dataset, dev_dataset, test_dataset, src_vocab, trg_vocab

def collate_fn(batch, src_vocab, trg_size):
    """
    Hàm collate để xử lý batch: padding các chuỗi nguồn và đích.

    :param batch: danh sách các mẫu từ dataset
    :param src_vocab: từ vựng nguồn
    :param trg_size: kích thước vector đích
    :return: dictionary chứa src, trg và file_paths đã được padding
    """

    src = [torch.tensor(src_vocab.tokens_to_indices(item['src'])) for item in batch]
    trg = [torch.tensor(item['trg'], dtype=torch.float32) for item in batch]
    file_paths = [item['file_path'] for item in batch]

    src_padded = pad_sequence(src, batch_first=True, padding_value=src_vocab.stoi[PAD_TOKEN])
    trg_padded = pad_sequence(trg, batch_first=True, padding_value=TARGET_PAD)

    return {'src': src_padded, 'trg': trg_padded, 'file_paths': file_paths}

def tokenize_line(line: str) -> list[str]:
    """
    Tách chuỗi thành danh sách token, giữ dấu câu như token riêng.
    """
    punctuation_spacing = re.compile(r'([.,!?;:()"\'-])')

    line = punctuation_spacing.sub(r' \1 ', line)
    tokens = [token for token in line.split() if token]
    return tokens

def make_data_iter(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """
    Trả về một DataLoader cho dataset.

    :param dataset: dataset chứa src và trg
    :param batch_size: kích thước batch
    :param shuffle: có xáo trộn dữ liệu trước mỗi epoch hay không
    :return: DataLoader
    """
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty. Cannot create DataLoader.")

    def collate(batch):
        return collate_fn(batch, dataset.src_vocab, dataset.trg_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn = collate
    )

class SignProdDataset(Dataset):
    """
    Định nghĩa dataset.
    """

    def __init__(self, path: str, trg_size: int, skip_frames: int = 1, max_sent_length: int = 300):
        """
        Khởi tạo dataset từ thư mục dữ liệu.

        :param path: đường dẫn đến thư mục chứa dữ liệu (.npy và .txt)
        :param trg_size: kích thước vector đích (bao gồm tất cả khớp + counter)
        :param skip_frames: số frame bỏ qua
        :param max_sent_length: độ dài tối đa của câu
        """

        self.path = path
        self.trg_size = trg_size
        self.skip_frames = skip_frames
        self.max_sent_length = max_sent_length
        self.examples = self._load_data()
        self.src_vocab = None

    def _load_data(self):
        examples = []
        
        # Kiểm tra xem đường dẫn có tồn tại không
        if not os.path.exists(self.path):
            print(f"Warning: Path {self.path} does not exist")
            return examples
            
        # Tìm tất cả các file .txt trong thư mục
        txt_files = glob.glob(os.path.join(self.path, "**", "*.txt"), recursive=True)
        
        if len(txt_files) == 0:
            # Thử tìm trong các folder con
            subfolders = [f.path for f in os.scandir(self.path) if f.is_dir()]
            print(f"Found {len(subfolders)} subfolders in {self.path}")
            
            for subfolder in subfolders:
                subfolder_txt_files = glob.glob(os.path.join(subfolder, "*.txt"))
                txt_files.extend(subfolder_txt_files)
                
            print(f"Found {len(txt_files)} .txt files in subfolders")
        else:
            print(f"Found {len(txt_files)} .txt files in {self.path}")
        
        for txt_file in txt_files:
            # Tên file cơ sở (không có phần mở rộng)
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            folder_path = os.path.dirname(txt_file)
            # Đường dẫn tương ứng đến file .npy
            npy_file = os.path.join(folder_path, f"{base_name}.npy")
            
            # Kiểm tra xem file .npy có tồn tại không
            if not os.path.exists(npy_file):
                print(f"Warning: Corresponding .npy file not found for {txt_file}")
                continue
            
            # Đọc nội dung từ file .txt
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    txt_content = f.read().strip()
                    src_line = tokenize_line(txt_content)
                    
                    # Bỏ qua nếu vượt quá độ dài tối đa
                    if len(src_line) > self.max_sent_length:
                        print(f"Skipping {base_name} due to excessive length: {len(src_line)} > {self.max_sent_length}")
                        continue
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
                continue
            
            # Đọc dữ liệu từ file .npy
            try:
                npy_data = np.load(npy_file)
                print(f"Loaded {base_name}.npy with shape {npy_data.shape}")
                
                # Nếu dữ liệu có shape (frame, trg_size), chỉ cần chia theo skip_frames
                frames = npy_data[::self.skip_frames]
                # Thêm epsilon nhỏ để tránh giá trị 0 chính xác
                frames = frames + 1e-8
                
                examples.append({
                    'src': src_line,
                    'trg': frames,
                    'file_path': base_name
                })
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
        
        print(f"Successfully loaded {len(examples)} examples")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

