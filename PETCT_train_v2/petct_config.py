import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--Task_name', type=str, default='PETCT_whole_body_task9_2channels')  # 任务名,也是文件名

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--model_name', type=str, default='smp.Unet')  # 模型框架
parser.add_argument('--encoder_name', type=str, default='timm-regnety_160')  # 编码结构  timm-regnety_160

# training hyper-parameters
parser.add_argument('--img_ch', type=int, default=2)
parser.add_argument('--output_ch', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=80)  # 总epoch

parser.add_argument('--num_epochs_decay', type=int, default=10)  # decay开始的最小epoch数
parser.add_argument('--decay_ratio', type=float, default=0.01)  # 0~1,每次decay到1*ratio
parser.add_argument('--decay_step', type=int, default=100)  # epoch

parser.add_argument('--batch_size', type=int, default=24)  # 训多少个图才回调参数
parser.add_argument('--batch_size_test', type=int, default=20)  # 测试时多少个,可以设很大,但结果图就会很小
parser.add_argument('--num_workers', type=int, default=8)

# 设置学习率
parser.add_argument('--lr', type=float, default=1e-5)  # 初始or最大学习率
parser.add_argument('--lr_low', type=float, default=1e-9)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)

parser.add_argument('--lr_cos_epoch', type=int, default=100)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
parser.add_argument('--lr_warm_epoch', type=int, default=0)  # warm_up的epoch数,一般就是10~20,为0或False则不使用

# optimizer param
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--augmentation_prob', type=float, default=0.5)  # 数据扩增的概率

parser.add_argument('--save_model_step', type=int, default=200)  # 多少epoch保存一次模型
parser.add_argument('--val_step', type=int, default=1)  #

parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=1)  # 跑第几折的数据

# data-parameters
parser.add_argument('--filepath_img', type=str, default=r"/data/newnas/MJY_file/h5_2d")
# 用于分折的csv表格
parser.add_argument('--csv_file', type=str,
                    default=r"/data/newnas/MJY_file/h5_csv/lesion_Info.csv")  # r"/data/newnas/ZSN/2022_miccai_petct/data/autoPETmeta.csv"
# result&save
parser.add_argument('--result_path', type=str, default=r"/data/newnas/MJY_file/MJY/Seg_result")  # 结果保存地址
parser.add_argument('--save_detail_result', type=bool, default=True)
parser.add_argument('--save_image', type=bool, default=True)  # 训练过程中观察图像和结果
parser.add_argument('--save_test_image', type=bool, default=True)  # 测试过程中观察图像和结果
parser.add_argument('--save_lastest_model', type=bool, default=False)  # 每个epoch保存一次model

# ==== 加入全身data到训练集中 ====
# 抽取全身的数据，间隔插入训练集内
parser.add_argument('--use_whole_body', type=bool, default=True)
parser.add_argument('--whole_body_scale', type=int, default=1)  # 全身图像的比例
parser.add_argument('--whole_body_epoch', type=int, default=4)  # 每次插入全身数据训练几个epoch
parser.add_argument('--all_csv_file', type=str,
                    default=r"/data/newnas/MJY_file/h5_csv/whole_body_Info_withoutNegetive.csv")
parser.add_argument('--lap_rate', type=float, default=0.25)  # 每个train_set和上一个的重叠率
parser.add_argument('--use_wb_test', type=bool, default=True)  # 用不用全身数据进行验证
parser.add_argument('--use_random_test', type=bool, default=True)  # 用不用全身数据的随机30个进行验证,False就是全部验证
parser.add_argument('--test_save_nii', type=bool, default=True)     # 测试保存nii
# more param
parser.add_argument('--mode', type=str, default='train', help='train/test/test_whole_body')  # 训练/测试
parser.add_argument('--test_model', type=str, default='best.pkl', help='best.pkl/lastest.pkl')  # 用最好的测试还是用最近的测试
parser.add_argument('--cuda_idx', type=int, default=0)  # 用几号卡的显存
parser.add_argument('--DataParallel', type=bool, default=False)  # 数据并行,开了可以用多张卡的显存,不推荐使用
parser.add_argument('--train_flag', type=bool, default=False)  # 训练过程中是否回测训练集,不测试会节省很多时间
parser.add_argument('--seed', type=int, default=2022)  # 随机数的种子点，一般不变
parser.add_argument('--TTA', type=bool, default=False)
config = parser.parse_args()
