import os
import h5py
import pandas as pd
import numpy as np

sep = os.sep
if __name__ == "__main__":
    h5_folder_path = r"/data/newnas/Yang_AutoPET/h5_2d"
    h5_folder_list = os.listdir(h5_folder_path)
    csv_folder_path = r"/data/newnas/Yang_AutoPET/h5_csv"
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    csv_lesion_saved_path = csv_folder_path + sep + "lesion_Info.csv"
    csv_all_saved_path = csv_folder_path + sep + "whole_body_Info_withoutNegetive.csv"
    csv_all = []
    csv_lesion = []
    for path in h5_folder_list:
        if path.split('_')[1] == 'NEGATIVE':
            continue
        file_path = os.path.join(h5_folder_path, path)
        for i in range(len(os.listdir(file_path))):
            #不要第一个和最后一个
            if i == 0 or i == len(os.listdir(file_path))-1:
                continue
            h5_file_path = os.path.join(file_path, str(i) + '.h5')
            h5_f = h5py.File(h5_file_path, "r")
            slice = np.array(h5_f["mask"], dtype='uint8')
            s = slice.sum()
            csv_all.append([path + sep + str(i) + ".h5", s])
            if s != 0:
                csv_lesion.append([path + sep + str(i) + ".h5", s])
            h5_f.close()

        print('patient: '+path+' successfully loaded')

    print("all patients are loaded")
    header_name = ['Path', 'Voxel_num']
    csv_data = pd.DataFrame(csv_lesion)
    csv_data.to_csv(csv_lesion_saved_path, header=header_name, index=None)
    csv_data = pd.DataFrame(csv_all)
    csv_data.to_csv(csv_all_saved_path, header=header_name, index=None)
