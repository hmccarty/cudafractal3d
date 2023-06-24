from numba import cuda
from math import log, log2
import math
import cv2
from scipy.signal import convolve2d
import numpy as np
np.set_printoptions(precision=15)

@cuda.jit
def mandelbrot(out, min, max, iter):
    idx, idy = cuda.grid(2)
    if idx > out.shape[0] or idy > out.shape[1]: return
    c = complex(min.real + (max.real - min.real) * idx / out.shape[0],
                min.imag + (max.imag - min.imag) * idy / out.shape[1])
    n, z = 0, complex(0.0, 0.0)
    while abs(z) <= 2 and n < iter:
        z = z*z + c
        n += 1
    out[idx, idy] = iter if n >= iter else n + 1 - log(log2(abs(z)))

def noise_driver(image, min, max):
    height, width = image.shape[:2]
    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]] 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = np.absolute(convolve2d(image_gray, M, mode='valid')[::2, ::2])
    sigma_shape = sigma.shape
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (width - 2) * (height - 2))
    idx = np.unravel_index(np.argmax(sigma), sigma.shape)
    idx = (idx[0]/sigma_shape[0] * height, idx[1]/sigma_shape[1] * width)
    return complex(min.real + (max.real - min.real) * idx[1] / width,
        min.imag + (max.imag - min.imag) * idx[0] / height)

if __name__ == "__main__":
    iters, width, height = 90.0, 1000, 800
    out = np.zeros((width, height), dtype=np.float64)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(out.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(out.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    iters_since_set, update_per_iters = 0, 15
    min, max = complex(-2.0, -1.5), complex(1.0, 1.5)
    center = complex(-0.71375, -0.24999)
    goal = complex(-0.71375, -0.24999)

    while True:
        mandelbrot[blockspergrid, threadsperblock](out, min, max, iters)

        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = (out.T * 255 / iters).astype(np.uint8)
        image[:, :, 1] = 255
        image[:, :, 2] = (255 * (out.T < iters)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # Calculate noise and set goal
        if iters_since_set > update_per_iters:
            goal = noise_driver(image, min, max)
            iters_since_set = 0
        else:
            iters_since_set += 1

        # Smoothly align center with goal
        center = complex((center.real * 0.9) + (goal.real * 0.1),
                         (center.imag * 0.9) + (goal.imag * 0.1))

        # Define new min and max by zooming into center
        zoom = 0.98
        min = complex(center.real - (center.real - min.real) * zoom,
            center.imag - (center.imag - min.imag) * zoom)
        max = complex(center.real + (max.real - center.real) * zoom,
            center.imag + (max.imag - center.imag) * zoom)

        # Show image
        cv2.imshow("Mandelbrot", image)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()