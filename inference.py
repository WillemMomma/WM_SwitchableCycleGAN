import argparse
import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import torch

from Model import Model
from Dataset_size import Dataset_test_M as Dataset_M
from Dataset_size import Dataset_test as Dataset
from torch.utils.data import DataLoader


def _save_dicom(fake, path, opt, type, alpha):
    fake = fake.cpu().detach().numpy()
    fake = fake.squeeze()

    if type == 'H':
        vmin = 400 - 1500 / 2
        vmax = 400 + 1500 / 2
    elif type == 'S':
        vmin = 50 - 120 / 2
        vmax = 50 + 120 / 2

    fake = np.clip(fake, a_min=vmin, a_max=vmax)
    
    # Read the original DICOM file to use as a template
    ds = pydicom.dcmread(path)
    ds.PixelData = fake.astype(np.int16).tobytes()

    # Update necessary DICOM metadata
    ds.Rows, ds.Columns = fake.shape

    save_path = os.path.join(opt.save_dir, opt.name, str(alpha), type)
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
    print(dataset_size)

    model = Model(opt, current_i=False)
    model.eval()
    for alpha in opt.alphas:
        print(f"Alpha: {alpha}")

        for i, data in enumerate(dataloader):
            model.set_input(data)
            real_H = model.real_H
            real_S = model.real_S
            # forward
            with torch.no_grad():
                fake_H = model.netG(real_S, alpha=float(alpha)).cpu()
                fake_S = model.netG(real_H, alpha=float(alpha)).cpu()

            fake_H = fake_H.squeeze()
            fake_S = fake_S.squeeze()

            # un-normalize for each volume
            fake_H = fake_H * data['stat_S'][1] + data['stat_S'][0]
            fake_S = fake_S * data['stat_H'][1] + data['stat_H'][0]

            # un-preprocessing
            fake_H = _unpreprocessing(fake_H)
            fake_S = _unpreprocessing(fake_S)

            # save results as DICOM
            _save_dicom(fake_H, model.path_S[0], opt, 'H', alpha)
            _save_dicom(fake_S, model.path_H[0], opt, 'S', alpha)

if __name__ == "__main__":
    main()
