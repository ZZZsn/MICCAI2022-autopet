import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import h5py
import matplotlib.pyplot as plt
import pylab
import itertools
from seg_code_2d.utils.misc import printProgressBar

sep = os.sep


def crop_image_256(ct, pet, GT):
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
    GT = GT[X_T:X_B, Y_L:Y_R]
    SUV = pet[X_T:X_B, Y_L:Y_R]

    return CT, SUV, GT, (X_T, X_B, Y_L, Y_R)


def get_path_list(patients_data):
    data_path_list = []
    for i in range(len(patients_data['study_location'])):
        data_path_list.append(folder_path + patients_data['study_location'][i][1:])
    return data_path_list


def get_dia_list(patients_data):
    diagnosis_list = [patients_data['diagnosis'][i] for i in range(len(patients_data['diagnosis']))]

    return diagnosis_list


# 所有索引从0开始
if __name__ == "__main__":
    csv_path = r"/data/newnas/ZSN/2022_miccai_petct/data/autoPETmeta.csv"
    folder_path = r"/data/newnas/ZSN/2022_miccai_petct/data/"
    h5_save_folder = r"/data/newnas/Yang_AutoPET/h5_2d_full_resolution"

    if not os.path.exists(h5_save_folder):
        os.makedirs(h5_save_folder)

    exsit_patient = os.listdir(h5_save_folder)  # 查看文件夹内已读入的
    id_to_load = int(len(exsit_patient)-1)  # 假定最后一位数据未完成，需要从此处开始
    # id_to_load = [111, 159, 218, 252, 417, 454, 492, 554, 741, 781, 834, 854, 868, 871, 873, 988]
    id_to_load = 2047
    patients_data = pd.read_csv(csv_path)
    path_list = get_path_list(patients_data)

    # path_list = path_list[id_to_load:]
    # patient_id = [i + 1 for i in range(len(path_list))]  # 从1开始定义病人id 暂时无用

    diagnosis_list = get_dia_list(patients_data)
    # 除去id值<现有文件夹的最大值的那些，直接进入目前最后一个读取任务
    for id, path in itertools.dropwhile(lambda a: a[0] < id_to_load, enumerate(path_list)):
    # for id in id_to_load:
    #     path = path_list[id]
    #     for id, path in enumerate(path_list):
        CT_nii_file = path + sep + 'CTres.nii.gz'
        SUV_nii_file = path + sep + 'SUV.nii.gz'
        mask_nii_file = path + sep + 'SEG.nii.gz'
        # 读取nii
        CT_nii = sitk.ReadImage(CT_nii_file)
        SUV_nii = sitk.ReadImage(SUV_nii_file)
        mask_nii = sitk.ReadImage(mask_nii_file)
        # 转换成数组形式
        CT = sitk.GetArrayFromImage(CT_nii).astype(np.float32)
        layer = CT.shape[0]
        SUV = sitk.GetArrayFromImage(SUV_nii).astype(np.float32)
        mask = sitk.GetArrayFromImage(mask_nii).astype(np.uint8)

        # plt.imshow(CT[150])
        # pylab.show()
        print('patient ' + str(id) + ' load data successfully')
        # 每层保存一个h5
        for i in range(CT.shape[0]):
            h5_save_file = h5_save_folder + sep + str(id) + '_' + diagnosis_list[id] + sep + str(
                i) + '.h5'  # 文件名 放在每个patient文件夹里
            if not os.path.exists(h5_save_folder + sep + str(id) + '_' + diagnosis_list[id]):
                os.makedirs(h5_save_folder + sep + str(id) + '_' + diagnosis_list[id])
            ct = CT[i]
            pet = SUV[i]
            GT = mask[i]
            # CT_256, SUV_256, GT_256 = crop_image_256(ct, pet, GT)
            h5_file = h5py.File(h5_save_file, 'w')
            h5_file['CT'] = ct
            h5_file['SUV'] = pet
            h5_file['mask'] = GT
            h5_file.close()
            printProgressBar(i + 1, layer)

        print('patient ' + str(id) + ' saved ' + str(layer) + ' h5 files successfully')

    print('ok')
