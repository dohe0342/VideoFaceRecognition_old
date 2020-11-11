PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python2.7

CUDA_LIB = /usr/local/cuda/lib64

BOOST_INC = /home/chge1532/boost_1_71_0/
#BOOST_INC = /usr/include
BOOST_LIB = /home/chge1532/boost_1_71_0/
#BOOST_LIB = /usr/lib
BOOST_NP = /home/chge1532/Boost.NumPy/build/lib/

TARGET = dboost

FGLAGS = --std=c++11

(TARGET).so: $(TARGET).o
	g++  -shared -Wl,--export-dynamic $(TARGET).o -L$(CUDA_LIB) -L$(BOOST_LIB) -L$(BOOST_NP) -l:libboost_numpy.so -l:libboost_python-py$(subst .,,$(PYTHON_VERSION)).so -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -lcuda -lcudart -o $(TARGET).so $(CFLAGS)

$(TARGET).o: $(TARGET).cu
	nvcc -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -lcuda -lcudart -Xcompiler -fPIC -c $(TARGET).cu $(CFLAGS) -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60



