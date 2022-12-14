import os
import pandas as pd
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.load(open(self.ann_path))
        self.examples = self.ann[self.split]
        self.masks = []
        self.reports = []

        for each in self.examples.keys():
            self.reports.append(self.examples[each]['En_Report'][:self.max_seq_length])
            self.masks.append([1]*len(self.reports[-1]))

    def __len__(self):
        return len(self.examples)

class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        # print("exam_dataset:", self.examples)
        # print("idx:", idx)
        case_ids = self.examples.keys()
        # print("ids:", case_ids) # ', 'case_10688'])
        # print("idss:",case_ids)
        case_id = list(case_ids)[idx]
        image_id = case_id
        # print("exam_dataset2:", self.examples)
        # self.ann['train'][example]['En_Report']
        print("self.examples:", self.examples[case_id]['Image_path'])
        image_path = self.examples[case_id]['Image_path']
        images = []
        for ind in range(len(image_path)):
            image = Image.open(os.path.join(self.image_dir, image_path[ind])).convert('RGB')
            if self.transform is not None:
                images.append(self.transform(image))
        images = torch.stack(images, 0)
        # report_ids = self.reports[idx]
        report_ids = image_id
        report_masks = self.masks[idx]
        seq_length = len(case_ids)
        sample = (image_id, images, report_ids, report_masks, seq_length)
        return sample


