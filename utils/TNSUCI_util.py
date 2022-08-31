from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import os
import warnings
import random
import numpy as np
import csv
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure

warnings.filterwarnings('ignore')
sep = os.sep
filesep = sep  # 设置分隔符


def char_color(s, front=50, word=32):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    new_char = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"  # 改变颜色的字符串
    return new_char


def array_shuffle(x, axis=0, random_state=2020):
    """
    对多维度数组，在任意轴打乱顺序
    :param x: ndarray
    :param axis: 打乱的轴
    :return:打乱后的数组
    """
    new_index = list(range(x.shape[axis]))
    random.seed(random_state)
    random.shuffle(new_index)
    x_new = np.transpose(x, ([axis] + [i for i in list(range(len(x.shape))) if i is not axis]))
    x_new = x_new[new_index][:]
    new_dim = list(np.array(range(axis)) + 1) + [0] + list(np.array(range(len(x.shape) - axis - 1)) + axis + 1)
    x_new = np.transpose(x_new, tuple(new_dim))
    return x_new


def get_filelist_frompath(filepath, expname, sample_id=None):
    """
    读取文件夹中带有固定扩展名的文件
    :param filepath:
    :param expname: 扩展名，如'h5','PNG'
    :param sample_id: 可以只读取固定患者id的图片
    :return: 文件路径list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.' + expname):
                id = int(file.split('.')[0])  # 以`.`为分隔符,然后第一个,也就得到id
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def get_wholebody_test_randomfilelist(csv_file, K=3, fold=1, random_state=2020, random_num=30):
    """
    为了在训练中得到全身验证的分数，随机采样病人提高效率
    :param csv_file:
    :param K: fold
    :param fold: fold
    :param:random_rate: 随机多少个数
    :return: 随机采样的set和原验证id给之后的全身测试用
    """
    csvlines = readCsv(csv_file)
    file_data = csvlines[1:]
    file_path = [i[0] for i in file_data]
    patient_num = [int(path.split('_')[0]) for path in file_path]
    patient_set = list(set(patient_num))  # 按病人分折  set:不重复的集合
    val_fold = []
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)  # 分K折
    for _, test_index in kf.split(patient_set):  # 获取k分割的索引值
        val_fold.append(np.array(patient_set)[test_index])
    test_id = val_fold[fold - 1]
    if random_num is not None:
        random_id = np.random.choice(test_id, random_num, replace=False).tolist()  # 随机选random_num个
    else:
        random_id = test_id # 不随机选，全部验证
    test_set = []
    # 取那些test_id内的图片
    for path in file_path:
        if int(path.split("_")[0]) in random_id:
            test_set.append(path)
    return test_set, random_id, test_id


def get_fold_filelist_4lesion(csv_file, K=3, fold=1, random_state=2020):
    """
        获取只含病例的病人分折结果
        输入只含病例的csv，返回所有只含病变的h5名,和test的id
        """
    csvlines = readCsv(csv_file)
    file_data = csvlines[1:]
    file_path = [i[0] for i in file_data]
    patient_num = [int(path.split('_')[0]) for path in file_path]
    patient_set = list(set(patient_num))  # 按病人分折  set:不重复的集合
    train_fold = []
    val_fold = []
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)  # 分K折
    for train_index, test_index in kf.split(patient_set):  # 获取k分割的索引值
        train_fold.append(np.array(patient_set)[train_index])
        val_fold.append(np.array(patient_set)[test_index])
    train_id = train_fold[fold - 1]
    test_id = val_fold[fold - 1]
    print('train_id:' + str(train_id) + '\nvalid_id:' + str(test_id))
    train_set = []
    test_set = []
    for path in file_path:
        if int(path.split("_")[0]) in train_id:
            train_set.append(path)
        else:
            test_set.append(path)
    return [train_set, test_set, test_id]


def get_fold_filelist_4all(csv_file, K=5, fold=1, extract_num=1, random_state=2020, lap_rate=0.25):
    """
        在全身h5文件中获取
        :param extract_num: 从全身的数据中取多少个病人
        返回一个装满分折病人的每一切片路径的列表
        """
    csvlines = readCsv(csv_file)
    file_data = csvlines[1:]
    file_path = [i[0] for i in file_data]
    patient_num = [int(path.split('_')[0]) for path in file_path]
    patient_set = list(set(patient_num))
    train_fold = []
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_set):  # 获取k分割的索引值
        train_fold.append(np.array(patient_set)[train_index])

    train_id = train_fold[fold - 1]
    lap = int(extract_num * lap_rate)  # 每个set重叠的数量
    block_num = len(train_id)

    train_num_limit = int(block_num / (extract_num - lap)) * (extract_num - lap)  # 限制取训练集的数量，保证均匀放置这些病人(不多拿不少拿)
    train_set_all = []
    # 全新（彩色）数据排列
    new_patient_list = []
    for i in range(0, train_num_limit, extract_num - lap):
        extra_list = train_id[i:i + extract_num - lap]
        new_patient_list.append(extra_list)
    num_round = len(new_patient_list)
    whole_body_set = []
    whole_body_set.append(new_patient_list[0])
    # 后几排的排列
    for i in range(1, num_round):
        last_set = whole_body_set[i - 1]
        lap_list = np.random.choice(last_set, lap, replace=False)  # 重复用的那部分的病人     # np array
        extra_list = new_patient_list[i]  # train id 也是np array
        extra_list = np.concatenate([extra_list, lap_list])
        whole_body_set.append(extra_list)
    # 第一排和最后一排的重叠
    last_set = whole_body_set[-1]
    lap_list = np.random.choice(last_set, lap, replace=False)
    extra_list = new_patient_list[0]
    extra_list = np.concatenate([extra_list, lap_list])
    whole_body_set[0] = extra_list  # 在最开始插入第一行的排列

    for ex_list in whole_body_set:
        train_set = []
        for h5_file in file_path:
            if int(h5_file.split("_")[0]) in ex_list:
                train_set.append(h5_file)
        train_set_all.append(train_set)
    return train_set_all


def save_4_nii(save_nii_path, pat_id, ct_whole, pet_whole, SR_whole, GT_whole):
    pred_nii_name = str(pat_id) + '_patient' + '_pred.nii.gz'
    gt_nii_name = str(pat_id) + '_patient' + '_gt.nii.gz'
    ct_nii_name = str(pat_id) + '_patient' + '_ct.nii.gz'
    pet_nii_name = str(pat_id) + '_patient' + '_pet.nii.gz'
    pred_nii_path = save_nii_path + sep + pred_nii_name
    gt_nii_path = save_nii_path + sep + gt_nii_name
    ct_nii_path = save_nii_path + sep + ct_nii_name
    pet_nii_path = save_nii_path + sep + pet_nii_name
    pred_save_sitk = sitk.GetImageFromArray(SR_whole.astype(np.uint8))
    gt_save_sitk = sitk.GetImageFromArray(GT_whole.astype(np.uint8))
    ct_save_sitk = sitk.GetImageFromArray(ct_whole)
    pet_save_sitk = sitk.GetImageFromArray(pet_whole)
    pred_save_sitk = sitk.Cast(pred_save_sitk, sitk.sitkUInt8)
    gt_save_sitk = sitk.Cast(gt_save_sitk, sitk.sitkUInt8)
    ct_save_sitk = sitk.Cast(ct_save_sitk, sitk.sitkFloat32)
    pet_save_sitk = sitk.Cast(pet_save_sitk, sitk.sitkFloat32)
    sitk.WriteImage(pred_save_sitk, pred_nii_path)
    sitk.WriteImage(gt_save_sitk, gt_nii_path)
    sitk.WriteImage(ct_save_sitk, ct_nii_path)
    sitk.WriteImage(pet_save_sitk, pet_nii_path)


def check_componnent_3d(mask, min_size=15, max_size=1e8):
    """
    检查全身分割结果，去掉不符合的病灶
    """
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        if np.sum(label_temp[:]) <= min_size:  # 体积太小
            mask[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
        if np.sum(label_temp[:]) >= max_size:  # 体积太大
            mask[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
        # 获取区域在z轴上的范围，如果不连续则去掉
        # label_temp_z = np.sum(np.sum(label_temp, axis=2), axis=1)
        # label_temp_z = label_temp_z > 0
        # if np.sum(label_temp_z) <= 1:  # z轴上不连续
        #     mask[label_temp] = 0
        #     continue
    return mask.astype(np.uint8)


def get_fold_filelist_sn(csv_file, K=3, fold=1, random_state=2020, validation=False, validation_r=0.2):
    """
       获取分折结果的API（）
       :param csv_file: 带有ID、CATE、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param random_state: 随机数种子
       :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
       :param validation_r: 抽取出验证集占训练集的比例
       :return: train和test的h5_list
       """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    print('header', header)
    nodules = csvlines[1:]
    # size = len(nodules)
    # print(size)
    data_id = [i[0] for i in nodules]
    # 对csv文件进行处理

    patient_id = []
    for file in data_id:  # file是字符串
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))

    patient_num = list(set(patient_id))  # 按病人分折  set:不重复的集合

    fold_train = []
    fold_test = []

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)  # 分K折
    for train_index, test_index in kf.split(patient_num):  # 获取k分割的索引值
        fold_train.append(np.array(patient_num)[train_index])
        fold_test.append(np.array(patient_num)[test_index])  # 得到train[array([123456]),array([123478])...],
        # test[array([78]),array([56])...]

    train_id = fold_train[fold - 1]
    test_id = fold_test[fold - 1]
    print('train_id:' + str(train_id) + '\nvalid_id:' + str(test_id))

    train_set = []
    test_set = []

    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in test_id:
            test_set.append(h5_file)
        else:
            train_set.append(h5_file)
    return [train_set, test_set]  # 找出相应病人的照片文件


def get_fold_filelist_train_some(csv_file, K=5, fold=1, extract_num=1, random_state=2020):
    """
       获取训练集里的设定例数的全身h5_list
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :param random_state: 随机数种子
       :return: train的h5_list
    """
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))  # 按病人分折

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    fold_train = []
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])

    train_id = fold_train[fold - 1]
    # 从训练集内随机抽取extract_num个id
    np.random.seed(random_state)  # 保证可重复性
    extract_train_id = np.random.choice(train_id, extract_num, replace=False)
    print('whole_body train_id:' + str(extract_train_id))

    train_set = []
    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in extract_train_id:
            train_set.append(h5_file)
    return train_set


def get_fold_filelist_train_all(csv_file, K=5, fold=1, extract_num=1, random_state=2020):
    """
       在训练集里按设定例数划分，取全身h5_list，组成一个集合
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :param random_state: 随机数种子
       :return: train的h5_list的集合
    """
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))  # 按病人分折

    fold_train = []

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])

    train_id = fold_train[fold - 1]
    train_num_limit = int(len(train_id) / extract_num) * extract_num  # 根据取的数目，限制训练集取的范围，即drop_last
    train_set_all = []
    for i in range(0, train_num_limit, extract_num):
        # 训练集内按间隔取
        extract_train_id = train_id[i:i + extract_num]
        train_set = []
        for h5_file in data_id:
            if int(h5_file.split('_')[0]) in extract_train_id:
                train_set.append(h5_file)
        train_set_all.append(train_set)
    return train_set_all


def save_nii(save_nii, CT_nii, save_path, save_mask=True):
    """
    保存nii
    :param save_nii: 需要保存的nii图像的array
    :param CT_nii: 配准的图像，用于获取同样的信息
    :param save_path: 保存路径
    :param save_mask: 保存的是否是mask，默认是True
    :return:
    """
    if save_mask:
        # 保存mask_nii
        save_sitk = sitk.GetImageFromArray(save_nii.astype(np.uint8))
        save_sitk.CopyInformation(CT_nii)
        save_sitk = sitk.Cast(save_sitk, sitk.sitkUInt8)
    else:
        # 保存img_nii
        save_sitk = sitk.GetImageFromArray(save_nii.astype(np.float))
        save_sitk.CopyInformation(CT_nii)
        save_sitk = sitk.Cast(save_sitk, sitk.sitkFloat32)

    sitk.WriteImage(save_sitk, save_path)
    print(save_path + ' processing successfully!')


def print_logger(logger, savepth):
    for index, key in enumerate(logger.keys()):
        figg = plt.figure()
        plt.plot(logger[key])
        figg.savefig(savepth + sep + key + '.PNG')
        plt.close()


def center_crop(imgs, body_mask, new_size):
    labeled_img, _ = measure.label(body_mask, connectivity=3, return_num=True)
    body_centroid = measure.regionprops(labeled_img)[0].centroid  # z,y,x
    img_crops = []
    for img in imgs:
        img_crop = img[:, int(body_centroid[1] - new_size / 2):int(body_centroid[1] - new_size / 2) + new_size,
                   int(body_centroid[2] - new_size / 2):int(body_centroid[2] - new_size / 2) + new_size]
        img_crops.append(img_crop)
    return img_crops
