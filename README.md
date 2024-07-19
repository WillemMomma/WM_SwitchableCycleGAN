# Switchable CycleGAN with AdaIN

This repository provides code for kernel switching using a CycleGAN network with AdaIN for style transfer. The goal is to visualize soft tissue from hard tissue kernel CT data.

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/WillemMomma/WM_SwitchableCycleGAN.git
cd WM_SwitchableCycleGAN
```

### Install Requirements
```bash
pip install -r requirements.txt
```

## Sample Data
The sample data is from the [Low Dose CT Grand Challenge][aapm link]. Download the data and place it in the `./dataset` directory for 2-domain and `./dataset_M` for 3-domain.

- [Sample Data][data link]

## Pretrained Models
Download the trained model weights and place them in the `./result/checkpoints/` directory.

- [Model Weights][model link]

## Training

### 2-Domain Switchable CycleGAN
Use `train.py` to train the 2-domain model.
```bash
python train.py
```

### 3-Domain Switchable CycleGAN
Use `train_M.py` to train the 3-domain model.
```bash
python train_M.py
```

## Testing

### 2-Domain Switchable CycleGAN
Use `inference.py` to test the 2-domain model.
```bash
python inference.py
```

### 3-Domain Switchable CycleGAN
Use `inference_M.py` to test the 3-domain model.
```bash
python inference_M.py
```

## Running Inference

To run inference on the trained 2-domain model, you can use the following command:

```bash
python3 inference.py --phase test --data_type Facial_bone --name last --load_epoch 191 --alphas 1.0 --data_dir ./dataset/Ct_face_test/ --save_dir ./result/checkpoints --gpu_ids -1
```

### Explanation of Parameters

- `--phase test`: Specifies the phase of the operation. For inference, this should be set to `test`.
- `--data_type Facial_bone`: Defines the type of data being processed. In this case, it is `Facial_bone`.
- `--name last`: The name of the experiment. This is used to locate the appropriate model checkpoints.
- `--load_epoch 191`: Indicates the specific epoch of the model to load for inference. In this example, epoch `191` is used.
- `--alphas 1.0`: Specifies the alpha values for the style transfer. Multiple values can be tested by providing a comma-separated list.
- `--data_dir ./dataset/Ct_face_test/`: The directory containing the input data for inference.
- `--save_dir ./result/checkpoints`: The directory where the results will be saved.
- `--gpu_ids -1`: Indicates which GPU to use for inference. Setting this to `-1` forces the code to run on the CPU.

## Acknowledgements
This code is based on the paper and original repository:
- [Continuous Conversion of CT Kernel using Switchable CycleGAN with AdaIN][paper link] (arXiv.org, Serin Yang et al.)

The authors implemented this code based on the [original CycleGAN codes][CycleGAN link].

[aapm link]: https://www.aapm.org/grandchallenge/lowdosect/
[data link]: https://drive.google.com/drive/folders/1SM0_-vJYB6xoDo0OtROdyYB7zfgJJmOw?usp=sharing
[model link]: https://drive.google.com/drive/folders/1xiVxB79IjPTipJwzkV1t_095X1kXk4kP?usp=sharing
[paper link]: https://arxiv.org/abs/2011.13150
[CycleGAN link]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
