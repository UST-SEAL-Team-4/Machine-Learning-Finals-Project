
import nibabel as nib
import numpy as np

# Load the NIfTI file
nifti_img = nib.load('Task2/sub-101/sub-101_space-T2S_CMB.nii.gz')
# nifti_img = nib.load('Task2/sub-101/sub-101_space-T2S_desc-masked_T2S.nii.gz')

# Get the data array from the image
data = nifti_img.get_fdata()

# Now you can access the values in the data array
# For example, if it's a 3D volume, you can access a specific voxel like this:
# voxel_value = data[x, y, z]
# Replace x, y, z with the coordinates of the voxel you're interested in

# Or you can iterate over all values
# x_num = 0
# y_num = 0
# z_num = 0
# print(data.shape[0], data.shape[1], data.shape[2])
# for x in range(data.shape[0]):
#     for y in range(data.shape[1]):
#         for z in range(data.shape[2]):
#             voxel_value = data[x, y, z]
#             # Do something with the voxel value
#             print(x_num, y_num, z_num, voxel_value)
#             z_num += 1
#         y_num += 1
#     x_num += 1

print(type(data))
print(data.shape)
# slice = data[:, :, 21]
# print("Slice ", slice)
# print(slice.max())
# print(slice.min())
i=0
for i in range(34):
    slice = data[:, :, i]
    # print("Slice ", slice)
    print(i, slice.max(), slice.min())
# # You can also use NumPy operations to analyze or manipulate the data
# import numpy as np
# mean_value = np.mean(data)
# max_value = np.max(data)
# min_value = np.min(data)
# print(mean_value)

x, y, z = np.where(data > None) if None is not None else np.where(data > 0)
print(x.shape, y.shape, z.shape)

# print(x, y, z)
# print("Headers: ", nifti_img.header)
# x, y, z = np.where(im
print(x, y, z)