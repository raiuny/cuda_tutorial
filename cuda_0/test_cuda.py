from concurrent.futures import thread
import time
import cv2
import numpy as np
from numba import cuda
import math

# GPU function
@cuda.jit
def process_gpu(img, channels):
    # 得到线程地址
    tx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    for c in range(channels):
        color = min(img[tx,ty][c]*2.0+30,255)
        img[tx,ty][c] = color


# CPU function
def process_cpu(img, dst):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = min(img[i,j][c]*2.0+30,255)
                dst[i,j][c] =color

if __name__=="__main__":
    img = cv2.imread('1.jpg')
    rows, cols, channels = img.shape
    print(img.shape)
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    # CPU 处理
    start_cpu = time.time()
    process_cpu(img, dst_cpu)
    end_cpu = time.time()
    print(f"CPU处理时间:{end_cpu-start_cpu}")

    # GPU 处理
    dImg = cuda.to_device(img)
    threadspreblock = (16,16)
    blockspergrid_x, blockspergrid_y = int(math.ceil(rows/threadspreblock[0])), int(math.ceil(cols/threadspreblock[1]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    cuda.synchronize()
    start_gpu = time.time()
    process_gpu[blockspergrid, threadspreblock](dImg, channels)
    cuda.synchronize()
    dst_gpu = dImg.copy_to_host()
    end_gpu = time.time()
    print(f"GPU处理时间:{end_gpu-start_gpu}")

    cv2.imwrite("result_cpu.jpg", dst_cpu)
    cv2.imwrite("result_gpu.jpg", dst_gpu)
    print("Done...")

