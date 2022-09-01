import SimpleITK
from UNet import predict
import os
import torch


class Petctsegmentationcontainer():
    def __init__(self):
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/'  # where to store the nii files
        self.model_path = '/opt/algorithm/checkpoints/best.pkl'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        # 把input文件从mha转到nii
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        # 把input文件从nii转到mha
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def predict(self, input_image: SimpleITK.Image):
        pass

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))


    def load_inputs(self):
        """
        read from /inputs
        """
        # 这个0什么意思呢？
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'SUV.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        write to /output
        """
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"),
                                os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def process(self):
        self.check_gpu()
        uuid = self.load_inputs()
        predict(self.model_path, self.nii_path, self.output_path)
        print('predict finished')
        self.write_outputs(uuid)


if __name__ == "__main__":
    Petctsegmentationcontainer().process()
