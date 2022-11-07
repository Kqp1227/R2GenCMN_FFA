import json
import os

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
        # self.ann = json.loads(open(self.ann_path, 'r').read())
        self.ann = json.load(open(self.ann_path))
        self.examples = self.ann[self.split]
        self.masks = []
        self.reports = []
        # iu_xray & mimic:
        # for i in range(len(self.examples)):
        #     self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
        #     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

        # ffa-ir:
        for each in self.examples.keys():
            self.reports.append(self.examples[each]['En_Report'][:self.max_seq_length])
            self.masks.append([1]*len(self.reports[-1]))

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        case_ids = self.examples.keys()
        # # print(case_ids)
        # print(idx)
        case_id = list(case_ids)[idx]
        image_id = case_id
        # example = self.examples[idx]
        # image_id = example['id']
        # print(self.examples[case_id])
        image_path = self.examples[case_id]['Image_path']
        images = []
        for ind in range(len(image_path)):
            image = Image.open(os.path.join(self.image_dir, image_path[ind])).convert('RGB')
            if self.transform is not None:
                images.append(self.transform(image))
        images = torch.stack(images, 0)
        report_ids = self.reports[idx]
        report_masks = self.masks[idx]

        seq_length = len(report_ids)
        sample = (image_id, images, report_ids, report_masks, seq_length)
        return sample