import datetime
import os
import time

import SimpleITK
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import ttach as tta
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from seg_code_2d.PETCT_train_v2.data_loader_petct import get_loader_2d_petct_h5
from seg_code_2d.loss.dice_loss import FocalTversky_loss
from seg_code_2d.loss.loss_weight import *
from seg_code_2d.loss.lovasz_losses import lovasz_hinge, binary_xloss
from seg_code_2d.utils.TNSUCI_util import *
from seg_code_2d.utils.scheduler import GradualWarmupScheduler
from seg_code_2d.utils.evaluation import *
from seg_code_2d.utils.misc import printProgressBar
from torchsummary import summary
import itertools
from whole_body_test import get_wholebody_metrics
from whole_body_test import get_wholebody_test_loader
from My_UNet import U_Net


class Solver(object):
    def __init__(self, config, train_loader, valid_loader):
        # Make record file
        self.record_file = config.record_file

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device.type + ':' + os.environ["CUDA_VISIBLE_DEVICES"])

        self.Task_name = config.Task_name

        # Data loader
        self.use_whole_body = config.use_whole_body
        self.whole_body_epoch = config.whole_body_epoch
        self.num_workers = config.num_workers
        self.train_list = config.train_list
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.wb_test_id = config.wb_test_id
        self.use_wb_test = config.use_wb_test
        self.random_test_id = config.random_test_id
        # 用不用随机全身提高效率
        self.random_val = config.use_random_test
        # self valid_list 只是显示用了多少的图片来验证
        if self.use_wb_test:
            self.valid_list = config.random_test_list

        else:
            self.valid_list = config.valid_list
        smp.unet
        # Models
        self.unet = None  # 模型，基本确定是使用unet结构
        self.model_name = config.model_name
        self.encoder_name = config.encoder_name
        self.optimizer = None
        self.img_ch = config.img_ch
        self.image_size = config.image_size
        self.output_ch = config.output_ch
        self.augmentation_prob = config.augmentation_prob

        # loss
        self.criterion = lovasz_hinge
        self.criterion1 = binary_xloss
        self.criterion2 = SoftDiceLoss()
        self.criterion4 = FocalTversky_loss()
        self.lw = AutomaticWeightedLoss(device=self.device, num=3)

        # Hyper-parameters
        self.lr = config.lr
        self.lr_low = config.lr_low
        if self.lr_low is None:
            self.lr_low = self.lr / 1e+6
            print("auto set minimun lr :", self.lr_low)

        # optimizer param
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.batch_size_test = config.batch_size_test

        # Step size
        self.save_model_step = config.save_model_step
        self.val_step = config.val_step
        self.decay_step = config.decay_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.save_image = config.save_image
        self.save_nii = config.test_save_nii
        self.save_test_image = config.save_test_image
        self.save_detail_result = config.save_detail_result
        self.log_dir = config.log_dir
        self.writer_4SaveAsPic = config.writer_4SaveAsPic
        self.log_pic_dir = config.log_pic_dir
        self.h5img_path = config.filepath_img
        # 设置学习率策略相关参数
        self.decay_ratio = config.decay_ratio
        self.num_epochs_decay = config.num_epochs_decay
        self.lr_cos_epoch = config.lr_cos_epoch
        self.lr_warm_epoch = config.lr_warm_epoch
        self.lr_sch = None  # 初始化先设置为None
        self.lr_list = []  # 临时记录lr

        # 其他参数
        self.DataParallel = config.DataParallel
        self.train_flag = config.train_flag
        self.save_lastest_model = config.save_lastest_model
        self.TTA = config.TTA
        if self.TTA:
            print('use TTA')  # 测试时扩增,一种提升结果的trick

        # 执行个初始化函数
        self.my_init()

    def myprint(self, *args):
        """Print & Record while training."""
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def my_init(self):
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.print_date_msg()
        self.build_model()

    def print_date_msg(self, train_list=None):
        if not self.use_whole_body:
            self.myprint("images count in train:{}".format(len(self.train_list)))
            self.myprint("images count in valid:{}".format(len(self.valid_list)))
        if train_list:
            self.myprint("images count in train:{}".format(len(train_list)))
            self.myprint("images count in valid:{}".format(len(self.valid_list)))

    def build_model(self):  # 构建网络,记得修改!
        """Build generator and discriminator."""
        # 用smp构建网络
        self.unet = eval(self.model_name)(encoder_name=self.encoder_name,
                                          encoder_weights='imagenet',
                                          in_channels=self.img_ch, classes=self.output_ch)
        # self.unet = U_Net(in_channel=2,num_classes=2)
        print("Bulid model with " + self.model_name + ',encoder:' + self.encoder_name)

        # 优化器修改
        self.optimizer = optim.Adam([{'params': self.unet.parameters()},
                                     {'params': self.lw.parameters(), 'lr': 1e-4, 'weight_decay': 0}],
                                    self.lr, (self.beta1, self.beta2))

        # lr schachle策略(要传入optimizer才可以)	学习率下降策略
        # 暂时的三种情况,(1)只用cos余弦下降,(2)只用warmup预热,(3)两者都用
        if self.lr_warm_epoch != 0 and self.lr_cos_epoch == 0:  # 只用预热
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr,然后在一定epoch内升回初始学习率lr
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=None)
            print('use warmup lr sch')
        elif self.lr_warm_epoch == 0 and self.lr_cos_epoch != 0:  # 只用余弦下降,在lr_cos_epoch内下降到最低学习率lr_low
            self.lr_sch = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         self.lr_cos_epoch,
                                                         eta_min=self.lr_low)
            print('use cos lr sch')
        elif self.lr_warm_epoch != 0 and self.lr_cos_epoch != 0:  # 都用
            self.update_lr(self.lr_low)
            scheduler_cos = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                           self.lr_cos_epoch,
                                                           eta_min=self.lr_low)
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=scheduler_cos)
            print('use warmup and cos lr sch')
        else:
            if self.lr_sch is None:
                print('use decay coded by dasheng')

        self.unet.to(self.device)
        if self.DataParallel:
            self.unet = torch.nn.DataParallel(self.unet)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.myprint(model)
        self.myprint(name)
        self.myprint("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable from tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, lr):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        """Convert tensor to img (numpy)."""
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""
        self.myprint('-----------------------%s-----------------------------' % self.Task_name)
        # 最优以及最新模型保存地址
        unet_best_path = os.path.join(self.model_path, 'best.pkl')
        unet_lastest_path = os.path.join(self.model_path, 'lastest.pkl')
        writer = SummaryWriter(log_dir=self.log_dir)

        # 判断是不是被中断的，如果是，那么就重载断点继续训练
        # 重新加载的参数包括：
        # 参数部分 1）模型权重；2）optimizer的参数，比如动量之类的；3）schedule的参数；4）epoch
        if os.path.isfile(unet_lastest_path):
            self.myprint('Reloading checkpoint information...')
            latest_status = torch.load(unet_lastest_path)
            self.unet.load_state_dict(latest_status['model'])
            self.lw.load_state_dict(latest_status['lw'])
            self.optimizer.load_state_dict(latest_status['optimizer'])
            self.lr_sch.load_state_dict(latest_status['lr_scheduler'])
            self.writer_4SaveAsPic = latest_status['writer_4SaveAsPic']
            print('restart at epoch:', latest_status['epoch'])
            best_unet_score = latest_status['best_unet_score']
            best_epoch = latest_status['best_epoch']
            epoch_start = latest_status['epoch']
            Iter = latest_status['Iter']
            #  断点的dataloader
            train_list_set_num = len(self.train_list)  # 共有多少列加了部分全身数据的train_list
            train_list_set_idx = int(epoch_start / self.whole_body_epoch)  # 判断轮到哪个train_list
            set_idx_now = train_list_set_idx % train_list_set_num  # 取余数判断该epoch属于0-7哪个set
            train_list = self.train_list[set_idx_now][0]
            self.train_loader = get_loader_2d_petct_h5(
                h5_list=train_list,
                image_size=self.image_size,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                mode='train',
                augmentation_prob=self.augmentation_prob)

        else:
            best_unet_score = 0.0
            best_epoch = 1
            epoch_start = 0
            Iter = 0

        valid_record = np.zeros((1, 8))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU]     # 记录指标
        # summary(self.unet, input_size=(2, 384, 384), batch_size=1, device="cuda")
        self.myprint('Training...')
        for epoch in range(epoch_start, self.num_epochs):
            tic = datetime.datetime.now()
            self.unet.train(True)
            epoch_loss = 0
            length = 0

            # 如果使用全身训练策略，重新构建train_data_loader
            if self.use_whole_body:
                if epoch % self.whole_body_epoch == 0:  # 每whole_body_epoch后（第wbe个epoch）换一次train_list
                    train_list_set_num = len(self.train_list)  # 共有多少列加了部分全身数据的train_list
                    train_list_set_idx = int(epoch / self.whole_body_epoch)  # 判断轮到哪个train_list
                    set_idx_now = train_list_set_idx % train_list_set_num  # 取余数判断该epoch属于0-7哪个set

                    # 每超过一轮则减 轮数*数据列数 ，确保一直在循环  这段有点一般 不用
                    # train_list_round = int(train_list_set_idx / train_list_set_num)
                    # train_list_set_idx -= train_list_round * train_list_set_num
                    # if train_list_set_idx >= train_list_set_num:
                    #     train_list_set_idx -= train_list_set_num
                    train_list = self.train_list[set_idx_now][0]
                    self.print_date_msg(train_list)

                    # if epoch < 10:
                    #     self.train_loader = get_loader_2d_petct_h5(
                    #         h5_list=train_list,
                    #         image_size=self.image_size,
                    #         batch_size=self.batch_size,
                    #         num_workers=self.num_workers,
                    #         mode='train',
                    #         shuffle=False,
                    #         augmentation_prob=self.augmentation_prob)
                    # else:

                    self.train_loader = get_loader_2d_petct_h5(
                        h5_list=train_list,
                        image_size=self.image_size,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        mode='train',
                        augmentation_prob=self.augmentation_prob)

            train_len = len(self.train_loader)

            for i, sample in enumerate(self.train_loader):
                # current_lr = self.optimizer.param_groups[0]['lr']
                # print(current_lr)
                # 调试
                # if i < 0.4*len(self.train_loader):
                #     continue
                (_, images, GT) = sample
                images = images.to(self.device, non_blocking=True)  # , non_blocking=True
                GT = GT.to(self.device, non_blocking=True)

                # 计算loss
                # SR : Segmentation Result
                SR = self.unet(images)
                # print('max logits = ',torch.max(SR))  # 查看是否真的是logits

                SR_probs = F.softmax(SR, dim=1)
                GT_onehot = F.one_hot(GT.squeeze(), num_classes=2).permute(0, 3, 1, 2).contiguous()
                SR_logits_flat = SR.view(SR.shape[0], -1)
                # SR_flat = SR_probs.view(SR_probs.shape[0], -1)
                GT_flat = GT_onehot.view(GT_onehot.shape[0], -1)
                # 把channel去掉，lovaszloss的预测输入是logits，[B,H,W],label是二值[B,H,W]
                # print(SR_logits_sq.shape)
                # print(GT_sqz.shape)

                # 计算loss
                # if epoch+1<=10:
                #     loss_lovz = 0
                # else:
                loss_lovz = self.criterion(SR_logits_flat, GT_flat)  # 输入就是要log its
                loss_sofdice = self.criterion2(SR_probs, GT_onehot)  # 输入两个矩阵[B,C,H,W]
                loss_bi_BCE = self.criterion1(SR_logits_flat, GT_flat)
                # loss_focal = self.criterion4(SR_flat, GT_flat)

                # 总loss
                # 先设置下各个loss的权重
                lovz_w = 1.0
                soft_dice_w = 1.0
                bi_BCE_w = 1.0
                # focal_w = 0.0
                # loss = soft_dice_w * loss_sofdice + lovz_w * loss_lovz + bi_BCE_w * loss_bi_BCE

                # 自动学习权重，这个本质上还是得确定大致量级后才能有效，目前感觉和全等差不多
                loss = self.lw(loss_lovz, loss_sofdice, loss_bi_BCE)
                weight = self.lw.get_weights()
                epoch_loss += float(loss)

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                length += 1
                Iter += 1
                writer.add_scalars('Loss', {'loss': loss}, Iter)

                if self.save_image and (i % 20 == 0):  # 20个batch图后保存一次png结果
                    if self.output_ch == 1:
                        images_all = torch.cat((images[:, 0:1, :, :], images[:, 1:2, :, :], SR_probs, GT), 0)
                    else:
                        images_all = torch.cat((images[:, 0:1, :, :], images[:, 1:2, :, :], SR_probs[:, 0:1, :, :],
                                                SR_probs[:, 1:2, :, :], GT), 0)
                    torchvision.utils.save_image(images_all.data.cpu(),
                                                 os.path.join(self.result_path, 'images', 'Train_%d_image.png' % i),
                                                 nrow=self.batch_size)

                # 储存loss到list并打印为图片
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer_4SaveAsPic['loss'].append(loss.data.cpu().numpy())
                self.writer_4SaveAsPic['loss_LOVAZ'].append(loss_lovz.data.cpu().numpy())
                self.writer_4SaveAsPic['loss_DICE'].append(loss_sofdice.data.cpu().numpy())
                self.writer_4SaveAsPic['loss_BCE'].append(loss_bi_BCE.data.cpu().numpy())
                self.writer_4SaveAsPic['lr'].append(current_lr)
                print_logger(self.writer_4SaveAsPic, self.log_pic_dir)

                # trainning bar
                print_content = 'batch_total_loss:' + str(loss.data.cpu().numpy()) + \
                                '  lovz:' + str(loss_lovz.data.cpu().numpy()) + \
                                '  BCE:' + str(loss_bi_BCE.data.cpu().numpy()) + \
                                '  dice:' + str(loss_sofdice.data.cpu().numpy()) + \
                                '  loss weight:[%.4f,%.4f,%.4f]' % (weight[0], weight[2], weight[1])
                # 顺序是lovas，bce，dice

                # print_content = 'batch_total_loss:' + str(loss.data.cpu().numpy()) + \
                #                 '  BCE:' + str(loss_bi_BCE.data.cpu().numpy()) + \
                #                 '  dice:' + str(loss_sofdice.data.cpu().numpy())
                printProgressBar(i + 1, train_len, content=print_content)

            # 计时结束
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)  # 小时和余数
            m, s = divmod(remainder, 60)  # 分钟和秒
            time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
            self.myprint(char_color(time_str))

            tic = datetime.datetime.now()

            epoch_loss = epoch_loss / length

            self.myprint('Epoch [%d/%d], Loss: %.4f lr: %f' % (epoch + 1, self.num_epochs, epoch_loss, current_lr))

            # 记录下lr到log里(并且记录到图片里)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_list.append(current_lr)
            writer.add_scalars('Learning rate', {'lr': current_lr}, epoch)

            # 保存lr为png
            figg = plt.figure()
            plt.plot(self.lr_list)
            figg.savefig(os.path.join(self.result_path, 'lr.PNG'))
            plt.close()
            figg, axis = plt.subplots()
            plt.plot(self.lr_list)
            axis.set_yscale("log")
            figg.savefig(os.path.join(self.result_path, 'lr_log.PNG'))
            plt.close()

            # ========================= 学习率策略部分 =========================
            # lr scha way 1:
            # 用上面定义的下降方式
            if self.lr_sch is not None:
                if (epoch + 1) <= (self.lr_cos_epoch + self.lr_warm_epoch):
                    self.lr_sch.step()
            # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
            # 超过num_epochs_decay后,每超过num_epochs_decay后阶梯下降一次lr
            if self.lr_sch is None:
                if ((epoch + 1) >= self.num_epochs_decay) and (
                        (epoch + 1 - self.num_epochs_decay) % self.decay_step == 0):
                    if current_lr >= self.lr_low:
                        self.lr = current_lr * self.decay_ratio
                        self.update_lr(self.lr)
                        self.myprint('Decay learning rate to lr: {}.'.format(self.lr))

            #  ========================= 验证 ===========================
            if (epoch + 1) % self.val_step == 0:
                if self.train_flag:
                    # Train
                    acc, SE, SP, PC, DC, IOU = self.test(mode='train')
                    self.myprint('[Train] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
                        acc, SE, SP, PC, DC, IOU))

                # Validation
                if self.use_wb_test:  # 是否使用随机病人全身图像来验证
                    acc, SE, SP, PC, DC, IOU = self.test_with_random_wb(mode='valid')
                else:
                    acc, SE, SP, PC, DC, IOU = self.test(mode='valid')

                valid_record = np.vstack((valid_record, np.array([epoch + 1, Iter, acc, SE, SP, PC, DC, IOU])))

                # TODO,以dsc作为最优指标
                unet_score = DC

                # 储存到tensorboard，并打印txt
                writer.add_scalars('Valid', {'Dice': DC, 'IOU': IOU}, epoch)
                self.myprint('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
                    acc, SE, SP, PC, DC, IOU))

                # 储存指标到pix
                self.writer_4SaveAsPic['score_val'].append([DC, IOU])
                print_logger(self.writer_4SaveAsPic, self.log_pic_dir)

                # 保存断点模型和其他参数，用于断点重训，每5个epoch保存一次
                # 发现断点重训的学习率不变，原因是用了预热策略，只用cos则不会
                if (epoch + 1) % 5 == 0:
                    lastest_state = dict(
                        model=self.unet.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        lr_scheduler=self.lr_sch.state_dict(),
                        epoch=epoch + 1,  # 此时学习率已经是改变过的，因此重载的epoch应该从下一个开始
                        Iter=Iter,
                        best_epoch=best_epoch,
                        best_unet_score=best_unet_score,
                        writer_4SaveAsPic=self.writer_4SaveAsPic,
                        lw=self.lw.state_dict()
                    )
                    torch.save(lastest_state, unet_lastest_path)

                # 最优模型保存，用于测试
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_state = dict(
                        model=self.unet.state_dict()
                    )
                    self.myprint('Best model in epoch %d, score : %.4f' % (best_epoch + 1, best_unet_score))
                    torch.save(best_state, unet_best_path)

                # save_record_in_xlsx
                if (True):
                    excel_save_path = os.path.join(self.result_path, 'record.xlsx')
                    record = pd.ExcelWriter(excel_save_path)
                    detail_result1 = pd.DataFrame(valid_record)
                    detail_result1.to_excel(record, 'valid', float_format='%.5f')
                    record.save()
                    record.close()

            # 规律性保存，一般设大一点的step不保存了
            if (epoch + 1) % self.save_model_step == 0:
                save_state = dict(
                    model=self.unet.state_dict(),
                    # optimizer=self.optimizer.state_dict(),
                    # lr_scheduler=self.lr_sch.state_dict(),
                    # epoch=epoch,
                    # best_epoch=best_epoch,
                    # best_unet_score=best_unet_score,
                    # writer_4SaveAsPic=self.writer_4SaveAsPic
                )
                torch.save(save_state, os.path.join(self.model_path, 'epoch%d_Testdice%.4f.pkl' % (epoch + 1, DC)))

            #
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "per epoch testing&vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
            print(char_color(time_str))

        self.myprint('Finished!')
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.myprint('Best model in epoch %d, score : %.4f' % (best_epoch + 1, best_unet_score))

    def test(self, mode='train', unet_path=None):
        """Test model & Calculate performances."""
        if not unet_path is None:
            if os.path.isfile(unet_path):
                best_status = torch.load(unet_path)
                self.unet.load_state_dict(best_status['model'], False)
                self.myprint('Best model is Successfully Loaded from %s' % unet_path)

        self.unet.train(False)
        self.unet.eval()
        csv_folder = r"/data/newnas/MJY_file/h5_csv/whole_body_test"
        h5_img_path = r"/data/newnas/MJY_file/h5_2d"
        if mode == 'train':
            data_loader = self.train_loader
            batch_size_test = self.batch_size
        elif mode == 'valid' or mode == 'val':
            data_loader = self.valid_loader
            batch_size_test = self.batch_size_test

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall) = TP/(TP+FN) 从正例中识别出正例的比率
        SP = 0.  # Specificity  = TN/(TN+FP) 从阴性中找出阴性的正确率
        PC = 0.  # Precision    = TP/(TP+FP) 识别正确的占模型识别为阳的比率
        DC = 0.  # Dice Coefficient = 2*（Y对应元素相乘y）/sum(Y)+sum(y)
        IOU = 0.  # IOU         = TP/(TP+FP+TP+FN-TP)
        length = 0

        # model pre for each image
        detail_result = []  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                (image_paths, images, GT) = sample

                # images_path = list(image_paths)
                images = images.to(self.device, non_blocking=True)
                GT = GT.to(self.device, non_blocking=True)

                # modelsize(self.unet, images)
                if not self.TTA:
                    # no TTA
                    SR = self.unet(images)
                    SR = F.sigmoid(SR)
                else:
                    # TTA
                    transforms = tta.Compose(
                        [
                            tta.VerticalFlip(),
                            tta.HorizontalFlip(),
                            tta.Rotate90(angles=[0, 180])
                            # tta.Scale(scales=[1, 2])
                            # tta.Multiply(factors=[0.9, 1, 1.1]),a
                        ]
                    )
                    tta_model = tta.SegmentationTTAWrapper(self.unet, transforms)
                    SR_mean = tta_model(images)
                    SR = F.sigmoid(SR_mean).float()
                if self.save_test_image and (length % 10 == 0):
                    if self.output_ch == 1:
                        images_all = torch.cat((images[:, 0:1, :, :], images[:, 1:2, :, :], SR, GT), 0)
                    else:
                        images_all = torch.cat(
                            (images[:, 0:1, :, :], images[:, 1:2, :, :], SR[:, 0:1, :, :], SR[:, 1:2, :, :], GT), 0)
                    torchvision.utils.save_image(images_all.data.cpu(),
                                                 os.path.join(self.result_path, 'images',
                                                              '%s_%d_image.png' % (mode, i)),
                                                 nrow=batch_size_test)
                SR_flat = SR.view(SR.shape[0], -1)
                GT_flat = GT.view(GT.shape[0], -1)  # 展平成batch个向量，放入计算真个batch的指标再求平均
                # SR = SR.data.cpu().numpy()
                # GT = GT.data.cpu().numpy()
                # for ii in range(SR.shape[0]):
                #     SR_tmp = SR[ii, :].reshape(-1)
                #     GT_tmp = GT[ii, :].reshape(-1)
                #
                #     tmp_index = images_path[ii].split('/')[-1]
                #     tmp_index = int(tmp_index.split('.')[0][:])
                #
                #     SR_tmp = torch.from_numpy(SR_tmp).to(self.device)
                #     GT_tmp = torch.from_numpy(GT_tmp).to(self.device)

                # acc, se, sp, pc, dc, _, _, iou = get_result_gpu(SR_tmp, GT_tmp) 	# 少楠写的
                result_tmp1 = get_result_gpu(SR_flat, GT_flat)
                result_tmp = np.array([i,
                                       result_tmp1[0],
                                       result_tmp1[1],
                                       result_tmp1[2],
                                       result_tmp1[3],
                                       result_tmp1[4],
                                       result_tmp1[7]])
                # print(result_tmp)
                acc += result_tmp[1]
                SE += result_tmp[2]
                SP += result_tmp[3]
                PC += result_tmp[4]
                DC += result_tmp[5]
                IOU += result_tmp[6]
                detail_result.append(result_tmp)

                length += 1
                # printProgressBar(length,len(data_loader))
        # 取平均
        accuracy = acc / length
        sensitivity = SE / length
        specificity = SP / length
        precision = PC / length
        disc = DC / length
        iou = IOU / length
        detail_result = np.array(detail_result)

        if (self.save_detail_result):  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            if mode == 'train':
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
            elif mode == 'test' and self.TTA:
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result_TTA.xlsx')
            else:
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result_test.xlsx')
            writer = pd.ExcelWriter(excel_save_path)
            detail_result = pd.DataFrame(detail_result)
            detail_result.to_excel(writer, mode, float_format='%.5f')
            writer.save()
            writer.close()
        return accuracy, sensitivity, specificity, precision, disc, iou

    def test_with_random_wb(self, mode, unet_path=None):
        """Test model & Calculate performances."""
        if not unet_path is None:
            if os.path.isfile(unet_path):
                best_status = torch.load(unet_path)
                self.unet.load_state_dict(best_status['model'], False)
                self.myprint('Best model is Successfully Loaded from %s' % unet_path)

        self.unet.train(False)
        self.unet.eval()
        csv_folder = r"/data/newnas/MJY_file/h5_csv/whole_body_test"
        h5_img_path = r"/data/newnas/MJY_file/h5_2d"
        # if mode == 'train':
        #     data_loader = self.train_loader
        #     batch_size_test = self.batch_size
        # elif mode == 'valid' or mode == 'val':
        #     data_loader = self.valid_loader
        batch_size_test = self.batch_size_test

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall) = TP/(TP+FN) 从正例中识别出正例的比率
        SP = 0.  # Specificity  = TN/(TN+FP) 从阴性中找出阴性的正确率
        PC = 0.  # Precision    = TP/(TP+FP) 识别正确的占模型识别为阳的比率
        DC = 0.  # Dice Coefficient = 2*（Y对应元素相乘y）/sum(Y)+sum(y)
        IOU = 0.  # IOU         = TP/(TP+FP+TP+FN-TP)
        length = 0

        # model pre for each image
        detail_result = []  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
        with torch.no_grad():
            # 用随机还是全部全身  self.wb_test_id/self.random_test_id
            for pat_id in self.random_test_id if self.random_val else self.wb_test_id:
                data_loader = get_wholebody_test_loader(pat_id,
                                                        csv_folder,
                                                        h5_img_path,
                                                        self.batch_size_test,
                                                        self.num_workers)
                gt_whole = []
                sr_whole = []
                for i, sample in enumerate(data_loader):
                    images, GT, _, _ = sample
                    # images_path = list(image_paths)
                    images = images.to(self.device, non_blocking=True)
                    GT = GT.to(self.device, non_blocking=True)
                    # modelsize(self.unet, images)
                    if not self.TTA:
                        # no TTA
                        SR = self.unet(images)
                        SR = F.softmax(SR, dim=1)
                    else:
                        # TTA
                        transforms = tta.Compose(
                            [
                                tta.VerticalFlip(),
                                tta.HorizontalFlip(),
                                tta.Rotate90(angles=[0, 180])
                                # tta.Scale(scales=[1, 2])
                                # tta.Multiply(factors=[0.9, 1, 1.1]),a
                            ]
                        )
                        tta_model = tta.SegmentationTTAWrapper(self.unet, transforms)
                        SR_mean = tta_model(images)
                        SR = F.sigmoid(SR_mean).float()
                    if self.output_ch == 1:
                        SR = SR.squeeze(1)
                    else:
                        _, SR = SR.max(1)
                    SR = SR.unsqueeze(1)
                    if self.save_test_image and (length % 20 == 0):
                        if self.output_ch == 1:
                            images_all = torch.cat((images[:, 0:1, :, :], images[:, 1:2, :, :], SR, GT), 0)
                        else:
                            images_all = torch.cat(
                                (images[:, 0:1, :, :], images[:, 1:2, :, :], SR, GT), 0)
                        torchvision.utils.save_image(images_all.data.cpu(),
                                                     os.path.join(self.result_path,
                                                                  'images',
                                                                  '%s_p%d_%d_image.png' % (mode, pat_id, i)),
                                                     nrow=batch_size_test)

                    SR = SR.cpu().numpy()
                    GT = GT.cpu().numpy()
                    sr_whole.extend(SR)
                    gt_whole.extend(GT)
                sr_whole = np.array(sr_whole)
                # sr_whole = check_componnent_3d(sr_whole)
                sr_whole = torch.from_numpy(sr_whole)
                gt_whole = torch.from_numpy(np.array(gt_whole))
                sr_whole = sr_whole.to(self.device)
                gt_whole = gt_whole.to(self.device)

                SR_flat = sr_whole.view(-1)
                GT_flat = gt_whole.view(-1)  # 展平成batch个向量，放入计算真个batch的指标再求平均
                # acc, se, sp, pc, dc, _, _, iou = get_result_gpu(SR_tmp, GT_tmp) 	# 少楠写的
                result_tmp1 = get_result_gpu(SR_flat, GT_flat)  # 整一个病人的分割指标
                result_tmp = np.array([i,
                                       result_tmp1[0],
                                       result_tmp1[1],
                                       result_tmp1[2],
                                       result_tmp1[3],
                                       result_tmp1[4],
                                       result_tmp1[7]])
                # print(result_tmp)
                acc += result_tmp[1]
                SE += result_tmp[2]
                SP += result_tmp[3]
                PC += result_tmp[4]
                DC += result_tmp[5]
                IOU += result_tmp[6]
                detail_result.append(result_tmp)

                length += 1
                # printProgressBar(length,len(data_loader))

        # 取平均
        accuracy = acc / length
        sensitivity = SE / length
        specificity = SP / length
        precision = PC / length
        disc = DC / length
        iou = IOU / length
        detail_result = np.array(detail_result)

        if (self.save_detail_result):  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            if mode == 'train':
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
            elif mode == 'test' and self.TTA:
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result_TTA.xlsx')
            else:
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result_test.xlsx')
            writer = pd.ExcelWriter(excel_save_path)
            detail_result = pd.DataFrame(detail_result)
            detail_result.to_excel(writer, mode, float_format='%.5f')
            writer.save()
            writer.close()
        return accuracy, sensitivity, specificity, precision, disc, iou

    def val_body_best_model(self, test_id, best_path):
        # 按病人来test
        self.unet.train(False)
        self.unet.eval()
        h5_img_path = self.h5img_path
        csv_folder = r"/data/newnas/MJY_file/h5_csv/whole_body_test"
        # acc = 0.  # Accuracy
        # SE = 0.  # Sensitivity (Recall) = TP/(TP+FN) 从正例中识别出正例的比率
        # SP = 0.  # Specificity  = TN/(TN+FP) 从阴性中找出阴性的正确率
        # PC = 0.  # Precision    = TP/(TP+FP) 正确的占模型识别正确的比率
        # DC = 0.  # Dice Coefficient = 2*（Y对应元素相乘y）/sum(Y)+sum(y)
        # IOU = 0.  # IOU         = TP/(TP+FP+TP+FN-TP)
        # length = 0
        best_state = torch.load(best_path)
        self.unet.load_state_dict(best_state["model"])
        print('testing patients amount', len(test_id))
        # detail_result = []
        dice_all = 0
        false_pos_all = 0
        false_neg_all = 0
        record = np.zeros(4)  # id dsc fp fn
        with torch.no_grad():
            for i, pat_id in enumerate(test_id):
                SR_whole = []
                GT_whole = []
                ct_whole = []
                pet_whole = []
                test_loader = get_wholebody_test_loader(pat_id, csv_folder, h5_img_path,
                                                        batch_size=self.batch_size_test,
                                                        num_workers=self.num_workers)
                length = 0
                for image, GT, ct, pet in test_loader:
                    # images_path = list(file_path)
                    image = image.to(self.device)
                    SR = self.unet(image)
                    SR = F.sigmoid(SR)
                    # SR = SR.view(SR.shape[2], -1)
                    if self.output_ch == 1:
                        SR = SR.squeeze(1)
                    else:
                        _, SR = SR.max(1)
                    GT = GT.squeeze(1)  # reshape 成(BHW)
                    SR_numpy = SR.data.cpu().numpy()
                    GT = GT.cpu().numpy()
                    ct = ct.cpu().numpy()
                    pet = pet.cpu().numpy()
                    SR_numpy[SR_numpy >= 0.5] = 1
                    SR_numpy[SR_numpy < 0.5] = 0
                    SR_whole.extend(SR_numpy)
                    GT_whole.extend(GT)
                    ct_whole.extend(ct)
                    pet_whole.extend(pet)
                    length += 1
                    printProgressBar(length,
                                     len(test_loader),
                                     content='predicting and stack')
                SR_whole = np.array(SR_whole)
                # 后处理 去除太小和太大的连通域
                # SR_whole = check_componnent_3d(SR_whole)

                GT_whole = np.array(GT_whole)
                ct_whole = np.array(ct_whole)
                pet_whole = np.array(pet_whole)
                # SR_whole = SR_whole.transpose(1, 2, 0)
                # GT_whole = GT_whole.transpose(1, 2, 0)
                # ct_whole = ct_whole.transpose(1, 2, 0)
                dice_sc, false_pos_vol, false_neg_vol = get_wholebody_metrics(pat_id, gt_array=GT_whole,
                                                                              pred_array=SR_whole)
                record = np.vstack((record, np.array([pat_id, dice_sc, false_pos_vol, false_neg_vol])))
                # 保存nii
                if self.save_nii:
                    save_nii_path = self.result_path + sep + 'whole_body_nii'
                    if not os.path.exists(save_nii_path):
                        os.makedirs(save_nii_path)
                    save_4_nii(save_nii_path, pat_id, ct_whole, pet_whole, SR_whole, GT_whole)
                dice_all += dice_sc
                false_pos_all += false_pos_vol
                false_neg_all += false_neg_vol
                content_str = 'patient %d DC: %f, FP: %f, FN: %f,' % (pat_id, dice_sc, false_pos_vol, false_neg_vol)
                printProgressBar(i + 1, len(test_id), content=content_str)
                print('\n')
            dice_ave = dice_all / len(test_id)
            FP_ave = false_pos_all / len(test_id)
            FN_ave = false_neg_all / len(test_id)
            # 把结果保存excel
            excel_save_path = os.path.join(self.result_path, 'wholebody_test_detail.xlsx')
            record = np.vstack((record, np.array([000, dice_ave, FP_ave, FN_ave])))[1:]  # all metrics
            ex_writer = pd.ExcelWriter(excel_save_path)
            detail = pd.DataFrame(record)
            header = ['pat_id', 'DICE', 'FP_vol', 'FN_vol']
            detail.to_excel(ex_writer, header=header, index=False)
            ex_writer.save()
            ex_writer.close()
        self.myprint('[Testing whole body]    Dice: %.4f, FP_vol: %f, FN_vol: %f' % (dice_ave, FP_ave, FN_ave))
        return dice_ave, FP_ave, FN_ave


def two_classes_seg(SR):
    SR_pred = torch.softmax(SR, dim=1)
    _, SR_class = SR_pred.max(1)
