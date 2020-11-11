import dboost
import ctypes
from ctypes import CDLL
import time
import numpy as np

db = ctypes.CDLL('./dboost.so')

goldenSize = 0
TILE_WIDTH = 16

#db.launch_kernel.restype = np.ctypeslib.ndpointer(dtype = np.float32)
#db.launch_kernel.argtypes = [ctypes.c_int, ctypes.c_int]

db.golden_init.argtypes = [ctypes.c_int]
db.golden_h2d.argtypes = [ctypes.c_int,
        np.ctypeslib.ndpointer(dtype = np.float32)]

db.frame_init.argtypes = [ctypes.c_int]
db.frame_h2d.argtypes = [ctypes.c_int,
        np.ctypeslib.ndpointer(dtype = np.float32)]

def bench(m, n):
    a_cpu = np.random.rand(m, 512).astype(np.float32)
    b_cpu = np.random.rand(512, n).astype(np.float32)

    ac = time.time()##
    c_cpu = np.matmul(a_cpu, b_cpu)
    c_cpu_maxidx = np.argmax(c_cpu, axis=1)
    c_cpu_maxval = np.max(c_cpu, axis=1)
    c_cpu = np.zeros(2 * m)
    for i in range(m):
        c_cpu[2 * i] = c_cpu_maxidx[i]
        c_cpu[2 * i + 1] = c_cpu_maxval[i]
    print(str(m) + ' by ' + str(n) + " CPU BENCH TIME = ", (time.time() - ac) * 1000, "ms")
    
    a_cpu = a_cpu.flatten()
    b_cpu = b_cpu.flatten()

    db.golden_init(ctypes.c_int(n))
    db.golden_h2d(ctypes.c_int(n), b_cpu)

    ac = time.time()##
    db.frame_h2d(ctypes.c_int(m), a_cpu)
    c_gpu = dboost.launch_kernel(m, n, 1)
    print(str(m) + ' by ' + str(n) + " GPU BENCH TIME T = ", (time.time() - ac) * 1000, "ms")

    print(c_cpu[0], c_cpu[1], int(c_gpu[0]), float(c_gpu[1]))
    print(c_cpu[2 * (m - 1)], c_cpu[2 * (m - 1) + 1], int(c_gpu[2 * (m - 1)]), float(c_gpu[2 * (m - 1) + 1]))

    print("error = ", np.sum(np.abs(np.subtract(c_cpu, c_gpu))) / np.sum(np.abs(c_cpu) + 0.00001) * 100)
    print("verification result = ", np.allclose(c_cpu, c_gpu, rtol=1e-03, atol=1e-03))

    ac = time.time()##
    db.frame_h2d(ctypes.c_int(m), a_cpu)
    c_gpu = dboost.launch_kernel(m, n, 0)
    print(str(m) + ' by ' + str(n) + " GPU BENCH TIME N = ", (time.time() - ac) * 1000, "ms")

    print(c_cpu[0], c_cpu[1], int(c_gpu[0]), float(c_gpu[1]))
    print(c_cpu[2 * (m - 1)], c_cpu[2 * (m - 1) + 1], int(c_gpu[2 * (m - 1)]), float(c_gpu[2 * (m - 1) + 1]))

    print("error = ", np.sum(np.abs(np.subtract(c_cpu, c_gpu))) / np.sum(np.abs(c_cpu) + 0.00001) * 100)
    print("verification result = ", np.allclose(c_cpu, c_gpu, rtol=1e-03, atol=1e-03))

#    db.golden_free()
#    db.frame_free()
#    db.result_free()


def init(_goldenSize, goldenVec, maxFrameSize, maxResultSize, tiled):
    global goldenSize, TILE_WIDTH
    if tiled:
        goldenSize = TILE_WIDTH * (int((_goldenSize - 1) / TILE_WIDTH) + 1)
    else:
        goldenSize = _goldenSize
    print("initializing facedb boost ", goldenSize)
    db.frame_init(maxFrameSize)
    db.result_init(maxResultSize)
    db.golden_init(ctypes.c_int(goldenSize))

    if tiled:
        goldenVec = goldenVec.reshape((512, -1))
        toAdd = goldenSize -  _goldenSize
        tempCol = np.zeros((512, 1)).astype(np.float32)
        for i in range(toAdd):
            goldenVec = np.hstack((goldenVec, tempCol))
        print('changed golden feature shape', goldenVec.shape)
        goldenVec = goldenVec.flatten().astype(np.float32)

    db.golden_h2d(ctypes.c_int(goldenSize), goldenVec)


def search_db(frameVec):
    global goldenSize
    frameSize = len(frameVec) / 512
    db.frame_h2d(ctypes.c_int(frameSize), frameVec)
    res_gpu = dboost.launch_kernel(frameSize, goldenSize, 0)
    return res_gpu


def search_db_tiled(frameVec):
    global goldenSize
    frameSize = len(frameVec) / 512
    db.frame_h2d(ctypes.c_int(frameSize), frameVec)
    res_gpu = dboost.launch_kernel(frameSize, goldenSize, 1)
    return res_gpu


def search_cpu(frameVec, goldenVec):
    frameSize = len(frameVec) / 512
    ac = time.time()
    frameVec = frameVec.reshape(frameSize, 512)
    goldenVec = goldenVec.reshape(512, len(goldenVec)/512)
    res_cpu = np.matmul(frameVec, goldenVec)
    res_cpu_maxidx = np.argmax(res_cpu, axis=1)
    res_cpu_maxval = np.max(res_cpu, axis=1)
    res_cpu = np.zeros(2 * frameSize)
    for i in range(frameSize):
        res_cpu[2 * i] = res_cpu_maxidx[i]
        res_cpu[2 * i + 1] = res_cpu_maxval[i]
    return res_cpu


db.frame_init(10000)# large amt preserved
db.result_init(32000000)

"""
bench(1, 10000)
bench(1, 100000)
bench(1, 1000000)

bench(10, 10000)
bench(10, 100000)
bench(10, 1000000)
"""

