import FaceRestoration
import cv2
import timeit

restore = FaceRestoration.FaceRestoration("models/model_256.engine")
img = cv2.imread("images/test.png")
img = cv2.resize(img, (512,512))
batch_img = np.repeat(img, 8, axis=0)
start = timeit.default_timer()
outputs = restore.infer(batch_img)
stop = timeit.default_timer()

print('Time: ', stop - start)
for output in outputs:
  cv2.imwrite("final.png", out) 
  print(out.shape)
