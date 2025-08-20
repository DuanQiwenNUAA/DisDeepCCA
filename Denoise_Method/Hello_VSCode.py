from matplotlib import pyplot as plt
import cv2

im_ascent=cv2.imread(r"D:\QQ\QQ_Data\anna.png",0)
print(im_ascent.dtype)
print(im_ascent.shape)

plt.imshow(im_ascent)
plt.show()

arr = cv2.imencode(".png",im_ascent)
arr[1].tofile(r"D:\QQ\QQ_Data\anna1.png")
