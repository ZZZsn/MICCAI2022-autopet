import torch
from torch.utils import data
from torchvision import transforms as T
import numpy as np
import os
from PIL import Image
import SimpleITK as sitk


def crop_image_256(ct, pet):
    y_sum = ct.sum(0)  # 生成一个行向量
    x_sum = ct.sum(1)  # 生成一个列向量
    min_y = y_sum.min()
    min_x = x_sum.min()
    # 可能某些切片是空的
    x_begin = 0
    x_end = 400
    y_begin = 0
    y_end = 400

    finding_end = 0
    for j, ct_num in enumerate(y_sum):
        if finding_end == 0:
            if ct_num > min_y:
                y_begin = j
                finding_end = 1
        if finding_end == 1:
            if ct_num == min_y:
                y_end = j
                break
    finding_end = 0
    for j, ct_num in enumerate(x_sum):
        if finding_end == 0:
            if ct_num > min_x:
                x_begin = j
                finding_end = 1
        if finding_end == 1:
            if ct_num == min_x:
                x_end = j
                break

    mid_point = [int((x_begin + x_end - 1) / 2), int((y_begin + y_end - 1) / 2)]
    if mid_point[0] not in range(100, 300) or mid_point[1] not in range(100, 300):
        mid_point = [200, 200]
    X_T = mid_point[0] - 128
    X_B = mid_point[0] + 128
    Y_L = mid_point[1] - 128
    Y_R = mid_point[1] + 128
    CT = ct[X_T:X_B, Y_L:Y_R]
    SUV = pet[X_T:X_B, Y_L:Y_R]

    return CT, SUV, (X_T, X_B, Y_L, Y_R)


class Test_Petct_NiiDataset(data.Dataset):
    def __init__(self, nii_path):
        """Initializes image paths and preprocessing module."""
        self.nii_path = nii_path
        CT_nii_file = os.path.join(nii_path, 'CTres.nii.gz')
        SUV_nii_file = os.path.join(nii_path, 'SUV.nii.gz')
        CT_nii = sitk.ReadImage(CT_nii_file)
        SUV_nii = sitk.ReadImage(SUV_nii_file)
        self.ct = sitk.GetArrayFromImage(CT_nii).astype(np.float32)
        self.layer = self.ct.shape[0]
        self.suv = sitk.GetArrayFromImage(SUV_nii).astype(np.float32)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        pet = self.suv[index]
        ct = self.ct[index]
        image_size = pet.shape[1]
        # crop and obtain 4 point
        ct, pet, point_tuple = crop_image_256(ct, pet)
        # 截断
        ct[ct < -1000] = -1000
        ct[ct > 1000] = 1000
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
        # 两个拼在一起，形成二通道
        pet = pet[:, :, np.newaxis]
        ct = ct[:, :, np.newaxis]
        image = np.concatenate((pet, ct), axis=2)
        # 确保大小正确+tensor化
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        Transform_img = []
        # Transform_img.append(T.ToTensor())
        # if image_size % 32 != 0:
        #     Transform_img.append(T.Resize([256, 256], Image.BILINEAR))
        #     Transform_img = T.Compose(Transform_img)
        #     image = Transform_img(image)
        image = image.type(torch.FloatTensor)
        return image, point_tuple

    def __len__(self):
        """Returns the total number of font files."""
        return self.layer


def get_loader_2d_petct(nii_path, batch_size=1, num_workers=0):
    """Builds and returns Dataloader."""
    dataset = Test_Petct_NiiDataset(nii_path)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return data_loader
