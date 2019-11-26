import scipy as sp
import scipy.misc
import numpy as np
from os import listdir, path, makedirs
import matplotlib.pyplot as plt


img_dir = "C:/Users/Yuu/Desktop/Doan2/yalefaces"
out_dir = "C:/Users/Yuu/Desktop/Doan2/_reconstruct"
eface_dir = "C:/Users/Yuu/Desktop/Doan2/_efaces"
compressed_dir = "C:/Users/Yuu/Desktop/Do An/_compressed"


img_names = listdir(img_dir)
img_list = list()
for fn in img_names:
    if (not path.isdir(path.join(img_dir,fn)) and not fn.startswith('.') and fn != "Thumbs.db"): 
        img = scipy.misc.imread(path.join(img_dir,fn),True)      
        img_list.append(img)
    
img_shape = img_list[0].shape

        
#unfold each image from 2D to 1D vector
#individual vector size is 77760x1 (243x320 = 77760) 
#Transposed to make 1D image vectors into columns 

imgs_mtrx=np.array([img.flatten() for img in img_list]).T
#
#
f = open("data.txt","w")
for i in range(imgs_mtrx.shape[0]):
    for j in range(imgs_mtrx.shape[1]):
        f.write(str(imgs_mtrx[i,j]) + " ")
        f.write("\n")
f.close()

mean_img = np.mean(imgs_mtrx,axis = 1)

mean_img_2d=mean_img.reshape(img_shape)


#subtracting mean from image matrix
for c_idx in range(imgs_mtrx.shape[1]):
    imgs_mtrx[:,c_idx] = imgs_mtrx[:,c_idx] - mean_img

#"A" is image matrix minus mean image
A=imgs_mtrx

#find eigenValues and eigenVectors for C
eigenvectors,eigenvalues,variance =np.linalg.svd(A, full_matrices=False)

efaces = eigenvectors[:,0:10] 
#
#
for i in range(efaces.shape[1]):
    f = open("eface_"+str(i)+".txt","w");
    for j in range(efaces.shape[0]):
        f.write(str(efaces[j,i]) + " ")
    f.close()

#calculate weights for each training image.
#projet the faces in eigenVector space
weights=np.dot(efaces.T,A)
#
#
for i in range(weights.shape[1]):
    f = open("w"+str(i)+".txt","w");
    for j in range(weights.shape[0]):
        f.write(str(weights[j,i]) + " ")
    f.close()
#
#
f = open("mean.txt","w")
for i in range(mean_img.shape[0]):
        f.write(str(mean_img[i,])+ " ")
f.close()

#reconstructed faces with the given weights and eigenfaces
recons_imgs = list()
for c_idx in range(imgs_mtrx.shape[1]):
    ri = mean_img + np.dot(efaces,weights[:,c_idx])
    recons_imgs.append(ri.reshape(img_shape))


#save mean and reconstructued imagesimport scipy.misc
if not path.exists(out_dir): makedirs(out_dir)
if not path.exists(eface_dir): makedirs(eface_dir)
for idx, img in enumerate(recons_imgs):
    sp.misc.imsave(path.join(out_dir,"img_"+str(idx)+".jpg"),img)
sp.misc.imsave(path.join(out_dir,"mean.jpg"),mean_img_2d)

for idx in range(efaces.shape[1]):
    sp.misc.imsave(path.join(eface_dir,"eface"+str(idx)+".jpg"),
                  efaces[:,idx].reshape(img_shape)) 
