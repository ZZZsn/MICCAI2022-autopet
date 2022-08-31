"""
用于输入PETCT双模态
"""
import datetime
import os

import torch
from torch.backends import cudnn

from seg_code_2d.PETCT_train_v2.data_loader_petct import get_loader_2d_petct_h5
from seg_code_2d.PETCT_train_v2.petct_solver import Solver
from seg_code_2d.utils.TNSUCI_util import *
from seg_code_2d.PETCT_train_v2.petct_config import config  # 参数设置在这里

sep = os.sep
if __name__ == '__main__':
    # step1: 设置随机数种子 -------------------------------------------------------
    seed = config.seed
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # step2: 设置各种文件保存路径 -------------------------------------------------------
    # 结果保存地址，后缀加上fold
    config.result_path = os.path.join(config.result_path,
                                      config.Task_name + '_fold' + str(config.fold_K) + '-' + str(config.fold_idx))
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')  # 在终端用 tensorboard --logdir=地址 指令查看指标
    config.log_pic_dir = os.path.join(config.result_path, 'logger_pic')
    config.writer_4SaveAsPic = dict(lr=[], loss=[], loss_DICE=[], loss_LOVAZ=[], loss_BCE=[], score_val=[])
    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        os.makedirs(config.model_path)
        os.makedirs(config.log_dir)
        os.makedirs(config.log_pic_dir)
        os.makedirs(os.path.join(config.result_path, 'images'))
    config.record_file = os.path.join(config.result_path, 'record.txt')
    f = open(config.record_file, 'a')
    f.close()

    # 训练时保存设置到txt
    if config.mode == 'train':
        print(config)
        f = open(os.path.join(config.result_path, 'config.txt'), 'w')
        for key in config.__dict__:
            print('%s: %s' % (key, config.__getattribute__(key)), file=f)
        f.close()

    # step3: GPU device -------------------------------------------------------
    cudnn.benchmark = True
    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)

    # step4: 数据集获取 -------------------------------------------------------
    # 改写了分折方式,确保按病人分折
    train, valid, _ = get_fold_filelist_4lesion(config.csv_file, K=config.fold_K, fold=config.fold_idx,
                                                random_state=seed)
    # train_folder_list = [config.filepath_img + sep + i for i in train]
    # valid_folder_list = [config.filepath_img + sep + i for i in valid]
    # train_list = []
    # valid_list = []
    # for path in train_folder_list:
    #     h5_list = os.listdir(path)
    #     for file in h5_list:
    #         train_list.append(path+sep+file)
    # for path in valid_folder_list:
    #     h5_list = os.listdir(path)
    #     for file in h5_list:
    #         valid_list.append(path+sep+file)
    train_list = [config.filepath_img + sep + i for i in train]
    valid_list = [config.filepath_img + sep + i for i in valid]

    # 全身训练策略
    if config.use_whole_body and config.mode == 'train':
        # 取同样的分折里的训练集的全身数据
        train_num = len(train_list)
        extra_num = np.ceil(train_num / 350 * config.whole_body_scale).astype(np.uint8)  # 向上取整，按一定比例得到要加入的例数
        extra_train_set = get_fold_filelist_4all(config.all_csv_file, K=config.fold_K, fold=config.fold_idx,
                                                 extract_num=extra_num, random_state=seed, lap_rate=config.lap_rate)
        train_list_add_set = []
        for num_set in range(len(extra_train_set)):
            extra_train = extra_train_set[num_set]
            extra_train_list = [config.filepath_img + sep + i for i in extra_train]
            in_num = int(np.ceil(len(extra_train_list) / train_num))
            # 等间隔插入原来训练数据中
            j = 1
            train_list_add = list(train_list)
            for i in range(int(len(extra_train_list) / in_num)):
                i = i * in_num
                train_list_add[j:j - 1] = extra_train_list[i:i + in_num]  # 在J处插入右边的赋值，右边是从i到i+插入数字（左闭右开），共in_num个，
                j += (in_num + 1)  # 然后给j加上in_num再加1,指向下一处插入的位置
            train_list_add_set.append([train_list_add])  # 为什么要加[]

        config.train_list = train_list_add_set  # 所有
    else:
        config.train_list = train_list

    random_numbers = 30 if config.use_random_test else None
    #用随机数量就设置为30，否之为None 全部用
    # 全身测试/验证用的随机id和所有id
    random_test_list,wb_random_test_id, wb_test_id = get_wholebody_test_randomfilelist(config.all_csv_file, K=config.fold_K,
                                                                      fold=config.fold_idx, random_state=seed, random_num=random_numbers )
    config.wb_test_id = wb_test_id
    config.random_test_id = wb_random_test_id
    config.valid_list = valid_list
    config.random_test_list = random_test_list      # 这两个list 只用来显示数量
    # 读取h5文件，交叉验证，只有训练和验证
    # 由于全身训练策略导致训练集的dataloader不断变化，因此在solver里会额外进行创建
    # train_loader = get_loader_2d_petct_h5(
    #     h5_list=train_list,
    #     image_size=config.image_size,
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    #     mode='train',
    #     augmentation_prob=config.augmentation_prob)
    train_loader = None
    valid_loader = get_loader_2d_petct_h5(
        h5_list=config.valid_list,
        image_size=config.image_size,
        batch_size=config.batch_size_test,
        num_workers=config.num_workers,
        mode='valid',
        augmentation_prob=0.0)

    # step4: 网络设置（包括学习率方法）、模型训练测试方案 -------------------------------------
    solver = Solver(config, train_loader, valid_loader)
    # step5: 训练or测试 -------------------------------------------------------
    if config.mode == 'train':

        solver.train()
    elif config.mode == 'test':  # 这里的测试本质上是单病灶区的2d验证，真正的验证和测试需要全身测试，额外写代码实现
        tic = datetime.datetime.now()  # 计时
        unet_best_path = os.path.join(config.model_path, 'best.pkl')
        print('=================================== test(val) ===================================')
        acc, SE, SP, PC, DC, IOU = solver.test_with_random_wb(mode='val', unet_path=unet_best_path)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
            acc, SE, SP, PC, DC, IOU))

        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "per epoch testing&vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
        print(char_color(time_str))
    elif config.mode == 'test_whole_body':
        tic = datetime.datetime.now()  # 计时
        best_path = os.path.join(config.model_path, 'best.pkl')
        lastest_path = os.path.join(config.model_path, 'lastest.pkl')
        test_path = os.path.join(config.model_path, config.test_model)
        if test_path == lastest_path:
            solver.myprint('============================= test(whole body with lastest) =============================')
        else:
            solver.myprint('============================= test(whole body with best) ==============================')
        # acc, SE, SP, PC, DC, IOU = solver.val_body_best_model(dataloader=test_loader, best_path=best_path)

        dice, FP, FN = solver.val_body_best_model(test_id=wb_test_id, best_path=test_path)

        print('[Testing whole body]    Dice: %.4f, FP_vol: %f, FN_vol: %f' % (dice, FP, FN))
        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "whold body testing cost Time %02d h:%02d m:%02d s" % (h, m, s)
        print(char_color(time_str))
