# convolutional neural networks (LeNet)
import pylab
from PIL import Image

img = Image.open(open('doc/images/3wolfmoon.jpg'))
img = numpy.asarray(img, dtype = 'float64') / 256.

img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 639, 516)
filtered_img = f(img_)

pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()

pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
