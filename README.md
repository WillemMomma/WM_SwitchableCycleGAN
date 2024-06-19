# TMI_SwitchableCycleGAN

## Paper
[Continuous Conversion of CT Kernel using Switchable CycleGAN with AdaIN][paper link] (arXiv.org, Serin Yang et al.)

[paper link]: https://arxiv.org/abs/2011.13150

## Code
We implemented this code based on the [original CycleGAN codes][CycleGAN link].

[CycleGAN link]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 

## Sample Data
[The sample data][data link] is from [Low Dose CT Grand Challenge][aapm link]. The data should be located on './dataset' and './dataset_M' for 2 domain and 3 domain, respectively.

[data link]: https://drive.google.com/drive/folders/143rA1SmTxMFmUGtPFIqZP3xOtNIxIbQ9?usp=sharing

[aapm link]: https://www.aapm.org/grandchallenge/lowdosect/

## Model
The trained model weights can be downloaded [here][model link]. The model should be located on './result/checkpoints/'

[model link]: https://drive.google.com/drive/folders/1xiVxB79IjPTipJwzkV1t_095X1kXk4kP?usp=sharing

## Train 
You can use train.py for 2 domain Switchable CycleGAN, and train_M.py for 3 domain Switchable CycleGAN.

## Test
You can use inference.py for 2 domain Switchable CycleGAN, and inference_M.py for 3 domain Switchable CycleGAN.

