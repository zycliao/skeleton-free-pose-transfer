import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from data_utils.mixamo_loader import MixamoDataset
from data_utils.rignet_loader import RignetDataset
from data_utils.amass_loader import AmassDataset


class MultiDataset(Dataset):
    # only used for training keypoint learning
    def __init__(self, amass_dir, mixamo_dir, rignet_dir,
                 smpl, part_augmentation=False, prob=(1/3, 1/3, 1/3), preload=None, single_part=True,
                 part_aug_scale=((0.5, 4), (0.6, 1), (0.3, 1.5)), simplify=True, new_rignet=True):
        # prob: probability of Amass data
        super(MultiDataset, self).__init__()
        if preload:
            self.amass, self.mixamo, self.rignet = preload
        else:
            if isinstance(part_augmentation, bool):
                p1 = p2 = p3 = part_augmentation
            else:
                p1, p2, p3 = part_augmentation
            self.amass = AmassDataset(amass_dir, smpl, part_augmentation=p1, simplify=simplify)
            self.mixamo = MixamoDataset(mixamo_dir, flag='train', part_augmentation=p2,
                                         single_part=single_part, part_aug_scale=part_aug_scale)
            if new_rignet:
                self.rignet = RignetDataset(rignet_dir, flag='humanoid_train_new')
            else:
                self.rignet = RignetDataset(rignet_dir, flag='humanoid_train')
        self.prob = prob

    def len(self):
        return 1000

    def database(self):
        return self.amass, self.mixamo, self.rignet

    def get(self, index):
        p = np.random.rand()
        if p <= self.prob[0]:
            return self.amass.get_uniform(np.random.randint(len(self.amass)))
        elif p <= self.prob[0] + self.prob[1] :
            return self.mixamo.get_uniform(np.random.randint(len(self.mixamo)))
        else:
            return self.rignet.get_uniform(np.random.randint(len(self.rignet)))

