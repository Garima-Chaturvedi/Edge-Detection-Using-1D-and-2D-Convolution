from PIL import Image
import numpy
import scipy
import math
import time
from scipy.misc import toimage

i = Image.open('lena_gray.png')
width, height = i.size

im = numpy.array(Image.open('lena_gray.png'))
im1 = Image.fromarray(im, 'L')
im1.show()

convx = numpy.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])

convy = numpy.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

convxr=numpy.array([1, 0, -1])

convxc=numpy.array([[1],
                    [2],
                    [1]])

convyr=numpy.array([1, 2, 1])

convyc=numpy.array([[1],
                    [0],
                    [-1]])

image = numpy.zeros(shape=(width + 2, height + 2), dtype=int)
image1 = numpy.array(image)

for m in range(0, width - 1):
    for n in range(0, height - 1):
        image1[m + 1, n + 1] = im[m, n]


def conv2d(w, h, image, conv):
    image2 = numpy.zeros(shape=(width, height), dtype=int)
    image2 = numpy.array(image2)
    for x in range(1, w - 2):
        for y in range(1, h - 2):
            image2[x-1, y-1] = (image[x - 1, y - 1] * conv[0, 0] + image[x - 1, y] * conv[0, 1] + image[x - 1, y + 1] * conv[0, 2] +
                                image[x, y - 1] * conv[1, 0] + image[x, y] * conv[1, 1] + image[x, y + 1] * conv[1, 2] +
                                image[x + 1, y - 1] * conv[2, 0] + image[x + 1, y] * conv[2, 1] + image[x + 1, y + 1] * conv[2, 2])
            if (image2[x, y] < 0):
                image2[x, y] =-image2[x,y]
    toimage(image2).show()
    return image2

G2d = numpy.zeros(shape=(width, height), dtype=int)
G2d = numpy.array(G2d)

time2dstart = time.time()

G2dx=conv2d(width + 2, height + 2, image1, convx)
G2dy=conv2d(width + 2, height + 2, image1, convy)

for x in range(0, width-1):
        for y in range(0, height-1):
            G2d[x,y]=math.sqrt(G2dx[x,y]*G2dx[x,y]+G2dy[x,y]*G2dy[x,y])
toimage(G2d).show()

time2dend = time.time()

def conv1d(w, h, image, convc, convr):
    image2 = numpy.zeros(shape=(w, h), dtype=int)
    image2 = numpy.array(image2)
    for x in range(1, w - 2):
        for y in range(1, h - 2):
            image2[x, y] = (image[x - 1, y] * convc[0, 0] + image[x, y] * convc[1,0] + image[x + 1, y] * convc[2, 0])
    image3 = numpy.zeros(shape=(width, height), dtype=int)
    image3 = numpy.array(image3)
    for x in range(1, w - 2):
        for y in range(1, h - 2):
            image3[x-1, y-1] = (image2[x, y-1] * convr[0] + image2[x, y] * convr[1] + image2[x, y+1] * convr[2])
            if (image3[x, y] < 0):
                image3[x, y] =-image3[x,y]
    toimage(image3).show()
    return image3

G1d = numpy.zeros(shape=(width, height), dtype=int)
G1d = numpy.array(G1d)

time1dstart = time.time()

G1dx=conv1d(width + 2, height + 2, image1, convxc, convxr)
G1dy=conv1d(width + 2, height + 2, image1, convyc, convyr)

for x in range(0, width-1):
        for y in range(0, height-1):
            G1d[x,y]=math.sqrt(G1dx[x,y]*G1dx[x,y]+G1dy[x,y]*G1dy[x,y])
toimage(G1d).show()
time1dend = time.time()

print('Conv2D function took %0.3f ms' %((time2dend-time2dstart)*1000.0))
print('Conv1D function took %0.3f ms' %((time1dend-time1dstart)*1000.0))

toimage(G2dx).save('G2dx.png')
toimage(G2dy).save('G2dy.png')
toimage(G2d).save('G2d.png')
toimage(G1dx).save('G1dx.png')
toimage(G1dy).save('G1dy.png')
toimage(G1d).save('G1d.png')
