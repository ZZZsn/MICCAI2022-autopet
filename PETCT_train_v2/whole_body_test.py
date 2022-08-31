import numpy as np
import pandas as pd
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import os
from seg_code_2d.utils.misc import printProgressBar
import h5py
import torch
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from data_preprocess.pre_processsing_2d_nii2h5 import crop_image_256

sep = os.sep
device = torch.device('cuda' if torch.cuda.is_available else "cpu")


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))  # 读nii
    pixdim = mask_nii.header['pixdim']
    voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000  # 获取nii中一个voxel的体积值
    return voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18  # 用什么邻接方式 18是面加边邻接
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    # conn_comp_t = torch.from_numpy(conn_comp)
    return conn_comp


def false_pos_pix(gt_array, pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    # 预测的连通分量上完全不与GT重叠的体积
    false_pos = 0
    gt_array = torch.from_numpy(gt_array).to(device)
    for idx in range(1, pred_conn_comp.max() + 1):
        comp_mask = np.isin(pred_conn_comp, idx)
        comp_mask = torch.from_numpy(comp_mask).to(device)
        if (comp_mask * gt_array).sum() == 0:
            false_pos = false_pos + comp_mask.sum()
        printProgressBar(idx + 1, pred_conn_comp.max() + 1, content='testing false_p')
    return false_pos


def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    # 完全假阴性  GT上完全不与pre重叠的体积
    false_neg = 0
    pred_array = torch.from_numpy(pred_array).to(device)
    for idx in range(1, gt_conn_comp.max() + 1):
        comp_mask = np.isin(gt_conn_comp, idx)
        comp_mask = torch.from_numpy(comp_mask).to(device)
        if (comp_mask * pred_array).sum() == 0:
            false_neg = false_neg + comp_mask.sum()

    return false_neg


def dice_score(mask1, mask2):
    # compute foreground Dice coefficient

    overlap = (mask1 * mask2).sum()
    sum = mask1.sum() + mask2.sum()
    dice_score = (2 * overlap / sum).item()
    return dice_score


def compute_metrics(nii_gt_path, gt_array, pred_array):
    # main function
    voxel_vol = nii2numpy(nii_gt_path)
    # 加速！！

    false_neg_vol = false_neg_pix(gt_array, pred_array) * voxel_vol

    false_pos_vol = false_pos_pix(gt_array, pred_array) * voxel_vol
    pred_array = torch.from_numpy(pred_array).to(device)
    gt_array = torch.from_numpy(gt_array).to(device)
    dice_sc = dice_score(gt_array, pred_array)

    return dice_sc, false_pos_vol.item(), false_neg_vol.item()


def get_wholebody_metrics(pat_id, gt_array, pred_array):
    nii_path = r'/data/newnas/ZSN/2022_miccai_petct/data'
    ori_csv = pd.read_csv(r"/data/newnas/ZSN/2022_miccai_petct/data/autoPETmeta.csv")
    ori_path_list = list(ori_csv['study_location'])

    nii_gt_path = ori_path_list[pat_id][1:]
    nii_gt_path = nii_path + nii_gt_path + 'SEG.nii.gz'
    nii_gt_path = plb.Path(nii_gt_path)

    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, gt_array, pred_array=pred_array)

    # csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    # csv_rows = [(nii_gt_path.name, dice_sc, false_pos_vol, false_neg_vol)]

    # with open(r"/data/newnas/Yang_AutoPET/metrics.csv", "w", newline='') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(csv_header)
    #     writer.writerows(csv_rows)
    return dice_sc, false_pos_vol, false_neg_vol


class get_dataset(data.Dataset):
    def __init__(self, pat_id, csv_folder, h5_folder_path):
        csv_name = str(pat_id) + '_pat.csv'
        csv = csv_folder + sep + csv_name
        self.file_list = pd.read_csv(csv)['Path']
        self.h5_folder_path = h5_folder_path

    def __getitem__(self, idx):
        file_path = self.h5_folder_path + sep + self.file_list[idx]
        h5_f = h5py.File(file_path, "r")
        ct = h5_f["CT"][()]
        pet = h5_f['SUV'][()]
        ct_copy = ct.copy()
        pet_copy = pet.copy()
        GT = h5_f['mask'][()]
        if ct.shape[0] !=256:
            ct, pet, GT, mark_point = crop_image_256(ct, pet, GT)
        GT = GT / 1.0
        GT = GT[:, :, np.newaxis]
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
        pet = pet[:, :, np.newaxis]
        ct = ct[:, :, np.newaxis]
        image = np.concatenate((pet, ct), axis=2)
        image = image.transpose(2, 0, 1)
        GT = GT.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        GT = torch.from_numpy(GT)
        Transform_img = []
        Transform_GT = []
        # if image.shape[1] % 32 != 0:
        #     Transform_img.append(T.Resize([384, 384], Image.BILINEAR))
        #     Transform_img = T.Compose(Transform_img)
        #     image = Transform_img(image)
        #     Transform_GT.append(T.Resize([384, 384], Image.NEAREST))
        #     Transform_GT = T.Compose(Transform_GT)
        #     GT = Transform_GT(GT)
        image = image.type(torch.FloatTensor)

        return image, GT, ct_copy, pet_copy

    def __len__(self):
        '''这波数据有多少'''
        return len(self.file_list)


def get_wholebody_test_loader(pat_id, csv_folder, h5_folder_path, batch_size, num_workers):
    wholebody_test_set = get_dataset(pat_id, csv_folder, h5_folder_path)
    test_loader = data.DataLoader(wholebody_test_set, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    return test_loader
