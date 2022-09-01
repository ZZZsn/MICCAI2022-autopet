import os
import SimpleITK as sitk
import numpy as np
import torch
import segmentation_models_pytorch as smp
from test_loader import get_loader_2d_petct
import nibabel as nib


class Net():
    def __init__(self):
        self.net = smp.Unet(encoder_name='timm-regnety_160', encoder_weights=None,
                            in_channels=2, classes=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = None
        self.nii_path = None
        self.batch_size = 1

    def data_preprocess(self, nii_path):
        self.dataloader = get_loader_2d_petct(nii_path, batch_size=self.batch_size, num_workers=0)
        self.nii_path = nii_path

    def inference(self, model_path, output_path):
        self.net.to(self.device)
        best_status = torch.load(model_path)
        self.net.load_state_dict(best_status['model'])
        self.net.eval()
        with torch.no_grad():
            sr_whole = []
            for i, sample in enumerate(self.dataloader):
                image = sample[0].to(self.device)
                cp = sample[1]  # corner point
                sr = self.net(image)
                sr = torch.sigmoid(sr)
                sr = sr.squeeze(1)  # 转为BHW
                sr = sr.cpu().numpy()
                # padding
                sr_400 = np.zeros([self.batch_size, 400, 400])
                sr_400[:, cp[0]:cp[1], cp[2]:cp[3]] = sr
                sr_whole.extend(sr_400)
            # sr_whole.reverse()
            sr_whole = np.array(sr_whole)  # (326,400,400)
            # sr_whole = sr_whole.transpose(1, 2, 0)  # (400,400,326)
            # thresholding
            sr_whole[sr_whole > 0.5] = 1
            sr_whole[sr_whole <= 0.5] = 0
            sr_whole = sr_whole.astype(np.uint8)

            # PT = nib.load(os.path.join(self.nii_path, "SUV.nii.gz"))
            # pet_affine = PT.affine
            # PT = PT.get_fdata()  # ??
            # mask_export = nib.Nifti1Image(sr_whole, pet_affine)
            save_sitk = sitk.GetImageFromArray(sr_whole.astype(np.uint8))
            pt = sitk.ReadImage(os.path.join(self.nii_path, "SUV.nii.gz"))
            save_sitk.CopyInformation(pt)
            save_sitk = sitk.Cast(save_sitk, sitk.sitkUInt8)
            sitk.WriteImage(save_sitk, os.path.join(output_path, "PRED.nii.gz"))
            print(os.path.join(output_path, "PRED.nii.gz"))
            print("done writing")


def predict(model_path, nii_path, output_path):
    net = Net()
    net.data_preprocess(nii_path)
    net.inference(model_path, output_path)
