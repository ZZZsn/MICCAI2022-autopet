import random
import torch
from torch.utils import data
from torchvision import transforms as T
from seg_code_2d.utils.img_mask_aug import *
import h5py
from PIL import Image

class ImageFolder_2d_petct_h5(data.Dataset):
    def __init__(self, h5list, image_size=512, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        # self.root = root

        # GT : Ground Truth
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.h5_paths = h5list

        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.norm = 'z-score'  # 标准化的三种选择： False-用全身统计结果做z标准化 'z-score'-z标准化 'normal'-归一化

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        h5_path = self.h5_paths[index]
        filename = h5_path.split('/')[-1]
        h5_data = h5py.File(h5_path, 'r')

        pet = h5_data['SUV'][()].astype(np.float32)
        # if pet.size == 0:
        #     print(h5_path + ' is error')
        #     print(h5_path + ' is error')
        #     print(h5_path + ' is error')
        #     print(h5_path + ' is error')
        ct = h5_data['CT'][()].astype(np.float32)
        GT_o = GT = h5_data['mask'][()].astype(np.int64)
        GT = GT[:, :, np.newaxis]
        # 截断
        ct[ct < -1000] = -1000
        ct[ct > 1000] = 1000
        if not self.norm:
            # 用全身统计结果做z标准化
            pet = (pet - 0.925055385) / 1.049767613
            ct = (ct - -45.52266631) / 236.3768854
        elif self.norm == 'z-score':
            # pet/ct标准化
            if pet.max() != 0:  # 防止空白层
                pet_mean = np.mean(pet[pet != 0])
                pet_std = np.std(pet[pet != 0])
                pet = (pet - pet_mean) / pet_std
            if ct.max() - ct.min() == 0:  # 防止空白层
                ct = (ct - ct.min())
            else:
                ct_mean = np.mean(ct[ct != -1000])
                ct_std = np.std(ct[ct != -1000])
                ct = (ct - ct_mean) / ct_std
        elif self.norm == 'normal':
            # pet/ct归一化
            if pet.max() != 0:  # 防止空白层导致nan
                pet_max = np.max(pet)
                pet_min = np.min(pet)
                pet = (pet - pet_min) / (pet_max - pet_min)
            if ct.max() - ct.min() == 0:  # 防止空白层导致nan
                ct = (ct - ct.min())
            else:
                ct_max = np.max(ct)
                ct_min = np.min(ct)
                ct = (ct - ct_max) / (ct_max - ct_min)
        # 两个拼在一起，形成二通道
        pet = pet[:, :, np.newaxis]
        ct = ct[:, :, np.newaxis]
        image = np.concatenate((pet, ct), axis=2)

        p_transform = random.random()  # 是否扩增
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # 扩增操作
            [image, GT] = data_aug_multimod(image, GT)  # 扩增操作

        # 确保大小正确+tensor化
        image = image.transpose(2, 0, 1)
        GT = GT.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        GT = torch.from_numpy(GT)
        Transform_img = []
        # Transform_img.append(T.ToTensor())
        if self.image_size % 32 != 0:
            Transform_img.append(T.Resize([256, 256], Image.BILINEAR))
            Transform_img = T.Compose(Transform_img)
            image = Transform_img(image)

        Transform_GT = []
        # Transform_GT.append(T.ToTensor())
        if self.image_size % 32 != 0:
            Transform_GT.append(T.Resize([256, 256], Image.NEAREST))
            Transform_GT = T.Compose(Transform_GT)
            GT = Transform_GT(GT)

        image = image.type(torch.FloatTensor)

        return h5_path, image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.h5_paths)


def get_loader_2d_petct_h5(h5_list, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4,
                           shuffle=True):
    """Builds and returns Dataloader."""
    dataset = ImageFolder_2d_petct_h5(h5list=h5_list, image_size=image_size, mode=mode,
                                      augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  drop_last=True, pin_memory=True)
    return data_loader
