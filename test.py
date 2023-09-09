import FaceRestoration
import cv2
import timeit

restore = FaceRestoration.FaceRestoration("../models/model_256.engine")
img = cv2.imread("../images/test.png")
img = cv2.resize(img, (256,256))
start = timeit.default_timer()
out = restore.infer(img)
stop = timeit.default_timer()

print('Time: ', stop - start)
cv2.imwrite("final.png", out) 
print(out.shape)
