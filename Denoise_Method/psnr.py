import cv2
import numpy as np
import math

#计算峰值信噪比
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

ori_img = cv2.imread(r"D:\Python_Projects\Denoise_Method\Denoise_Method\lena.png") #原始图片
n_img= cv2.imread(r"D:\Python_Projects\Denoise_Method\Denoise_Method\noise_lena.png") #加噪后的图片
den_img= cv2.imread(r"D:\Python_Projects\Denoise_Method\Denoise_Method\denoise_lena.png") #去噪后的图片

print('加噪后信噪比：',psnr(ori_img,n_img))
print('去噪后信噪比：',psnr(ori_img,den_img))

