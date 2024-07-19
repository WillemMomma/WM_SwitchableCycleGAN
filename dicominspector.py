import pydicom

path_h = "/Users/willemmomma/Documents/Work/AmsterdamUMC/CycleGAN/WM_SwitchableCycleGAN/result/checkpoints/last/1.0/H/CT_0.dcm"
path_s = "/Users/willemmomma/Documents/Work/AmsterdamUMC/CycleGAN/WM_SwitchableCycleGAN/result/checkpoints/last/1.0/S/CT_0.dcm"

# inspect the dicom file
ds_h = pydicom.dcmread(path_h)
ds_s = pydicom.dcmread(path_s)

print("--------- Dicom inspection for H --------------")
print(ds_h)
print("--------- DONE --------------")
print("--------- Dicom inspection for S --------------")
print(ds_s)
print("--------- DONE --------------")