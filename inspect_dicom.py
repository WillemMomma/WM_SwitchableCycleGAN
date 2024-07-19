import pydicom
import numpy as np

def inspect_dicom(path):
    # Read the dicom file
    ds = pydicom.dcmread(path)

    # Print the dicom file details
    # print(f"--------- Dicom inspection for {path} --------------")
    # print(ds)

    # Print the pixel data
    print(f"--------- Pixel data for {path} --------------")
    print(ds.pixel_array)

    # Print the shape of the image
    print(f"--------- Image shape for {path} --------------")
    print("Shape: ", ds.pixel_array.shape)

    # Print the data type of the pixel array
    print(f"--------- Data type for {path} --------------")
    print("Data type: ", ds.pixel_array.dtype)

    # Print the max and min pixel values
    print(f"--------- Range pixel value for {path} --------------")
    print("Max value: ", np.max(ds.pixel_array))
    print("Min value: ", np.min(ds.pixel_array))

    # Print the mean and standard deviation of pixel values
    print(f"--------- Pixel value statistics for {path} --------------")
    print("Mean value: ", np.mean(ds.pixel_array))
    print("Standard deviation: ", np.std(ds.pixel_array))

def main():
    path_s = "dataset/Ct_face_test/samples/S/CT_10.dcm"
    path_h = "dataset/Ct_face_test/samples/H/CT_10.dcm"
    path_adjusted = "result/checkpoints/last/0.7/S/CT_10.dcm"

    inspect_dicom(path_s)
    inspect_dicom(path_h)
    inspect_dicom(path_adjusted)

if __name__ == "__main__":
    main()