from numba import cuda
import math 
import cv2
import numpy as np
np.set_printoptions(precision=15)

@cuda.jit
def mandelbulb(out, power, min, max, max_iter):
    idx, idy, idz = cuda.grid(3)
    if idx > out.shape[0] or idy > out.shape[1] or idz > out.shape[2]: return
    c = cuda.float64x3(((max[0] - min[0]) * idx / out.shape[0]) + min[0], 
                       ((max[1] - min[1]) * idy / out.shape[1]) + min[1],
                       ((max[2] - min[2]) * idz / out.shape[2]) + min[2])
    n, v, r = 0, cuda.float64x3(0.0, 0.0, 0.0), 0.0
    while r <= 4.0 and n < max_iter:
        rn = math.pow(r, power)
        phi = math.atan2(v.y, v.x)
        theta = math.atan2(math.sqrt(v.x*v.x + v.y*v.y), v.z)
        v = cuda.float64x3(
            rn * (math.sin(power * theta) * math.cos(power * phi)),
            rn * math.sin(power * theta) * math.sin(power * phi),
            rn * math.cos(power * theta))
        v = cuda.float64x3(v.x + c.x, v.y + c.y, v.z + c.z) 
        r = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
        n += 1
    
    out[idx, idy, idz] = n

@cuda.jit
def tinyrtx(out, space):
    idx, idy = cuda.grid(2)
    if idx > out.shape[0] or idy > out.shape[1]: return
    for i in range(space.shape[2]):
        if space[idx, idy, i] == 90:
            out[idx, idy] = i
            return
    out[idx, idy] = 0

if __name__ == "__main__":
    iters, width, height, depth = 90.0, 600, 600, 600
    out = np.zeros((width, height, depth), dtype=np.float64)
    
    threadsperblock = (8, 8, 8)
    blockspergrid_x = int(np.ceil(out.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(out.shape[1] / threadsperblock[1]))
    blockspergrid_z = int(np.ceil(out.shape[2] / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _threadsperblock = (8, 8)
    _blockspergrid_x = int(np.ceil(out.shape[0] / _threadsperblock[0]))
    _blockspergrid_y = int(np.ceil(out.shape[1] / _threadsperblock[1]))
    _blockspergrid = (_blockspergrid_x, _blockspergrid_y)

    iters_since_set, update_per_iters = 0, 10
    min, max = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)

    while True:
        mandelbulb[blockspergrid, threadsperblock](out, 8, min, max, iters)

        space = out.copy()
        out = np.zeros((width, height), dtype=np.float64)
        tinyrtx[_blockspergrid, _threadsperblock](out, space)

        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = (out.T * 255 / iters).astype(np.uint8)
        image[:, :, 1] = 255
        image[:, :, 2] = (255 * (out.T < iters)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # Show image
        cv2.imshow("Mandelbulb", image)
        if cv2.waitKey(0) == ord('q'):
            break
    
    cv2.destroyAllWindows()