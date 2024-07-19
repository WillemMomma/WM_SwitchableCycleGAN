import argparse
import os
import pydicom
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from Model import Model
from Dataset_size import Dataset_test_M as Dataset_M
from Dataset_size import Dataset_test as Dataset
from torch.utils.data import DataLoader


def _save_dicom(fake, path, opt, alpha):
    fake = fake.cpu().detach().numpy()
    fake = fake.squeeze()

    # Read the original DICOM file to use as a template
    ds = pydicom.dcmread(path)
    
    # Ensure pixel data is uncompressed
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()

    vmin = -20
    vmax = 120

    # Clip the pixel values to the defined range
    fake = np.clip(fake, a_min=vmin, a_max=vmax)
    
    # Normalize the pixel values to fit within the DICOM pixel data range
    fake = (fake - vmin) / (vmax - vmin)
    fake = (fake * 4095).astype(np.uint16)  # Assuming 12-bit data (0-4095)
    
    ds.PixelData = fake.tobytes()

    # Update necessary DICOM metadata
    ds.Rows, ds.Columns = fake.shape
    
    # Change PatientName and PatientID to make the DICOM unique
    ds.PatientName = f"Alpha = {alpha} {datetime.now().strftime('%Y/%m/%d')}"
    ds.PatientID = f"alpha{alpha}_{datetime.now().strftime('%Y_%m_%d')}"

    save_path = os.path.join(opt.save_dir, opt.name, str(alpha), 'S')
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, os.path.basename(path))
    
    ds.save_as(save_file)


def _unpreprocessing(image):
    output = image
    mu_h2o = 0.0192
    output = (output - mu_h2o) * 1000 / mu_h2o     
    return output

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--data_type', type=str, default='Facial_bone', help='Facial_bone or Head')
    parser.add_argument('--name', type=str, default='last', help='name of the experiment')
    parser.add_argument('--load_epoch', type=int, default=191, help='if specified, print more debugging information')
    parser.add_argument('--load_best_model', type=int, default=0, help='True: save only best model, False: save model at each best epoch')
    parser.add_argument('--alphas', type=str, default='0.0,0.5,1.0', help='alpha values to be tested')
    parser.add_argument('--data_size', type=str, default='whole', help='whole, half, quarter')

    parser.add_argument('--isTrain', action='store_true', help='Train or Test')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--shuffle', type=int, default=0, help='shuffle')

    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--gpu_parallel', type=int, default=0, help='Parallel or Not for Inference')  
    parser.add_argument('--gpu_parallel_train', type=int, default=1, help='Parallel learning or Not for Training')
    parser.add_argument('--save_dir', type=str, default='./result/checkpoints', help='directory for model save')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='directory for model save')

    return parser.parse_args()

def main():
    opt = parse_arguments()

    temp = []
    for i in range(len(opt.alphas.split(','))):
        temp.append(opt.alphas.split(',')[i])
    opt.alphas = temp
    print(opt)

    # Update to check for the actual directory structure
    if 'dataset_M' in opt.data_dir.split('/'):
        dataset = Dataset_M(opt)
    elif 'dataset' in opt.data_dir.split('/'):
        dataset = Dataset(opt)
    else:
        raise ValueError(f"Unrecognized data directory structure: {opt.data_dir}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    dataset_size = len(dataset)
    print("Dataset Size: ", dataset_size)

    model = Model(opt, current_i=False)
    model.eval()
    for alpha in opt.alphas:
        print(f"Alpha: {alpha}")

        for i, data in enumerate(tqdm(dataloader, desc=f"Processing alpha {alpha}")):
            model.set_input(data)
            real_H = model.real_H

            # forward
            with torch.no_grad():
                fake_S = model.netG(real_H, alpha=float(alpha)).cpu()

            fake_S = fake_S.squeeze()

            # un-normalize for each volume
            fake_S = fake_S * data['stat_H'][1] + data['stat_H'][0]

            # un-preprocessing
            fake_S = _unpreprocessing(fake_S)

            # save results as DICOM
            _save_dicom(fake_S, model.path_H[0], opt, alpha)

if __name__ == "__main__":
    main()
