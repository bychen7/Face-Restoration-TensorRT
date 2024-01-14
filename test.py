import FaceRestoration
import cv2
import timeit
import numpy as np

restore = FaceRestoration.FaceRestoration("models/model.engine")
img = cv2.imread("images/test.png")
img = cv2.resize(img, (512,512))
batch_img = np.repeat(img[np.newaxis, :, :, :], 8, axis=0)
start = timeit.default_timer()
print(batch_img.shape)
outputs = restore.infer(batch_img)
stop = timeit.default_timer()

print('Time: ', stop - start)
for output in outputs:
  cv2.imwrite("final.png", out)
  print(out.shape)
