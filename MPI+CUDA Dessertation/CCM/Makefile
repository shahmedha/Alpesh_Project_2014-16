DEBUG = 
OBJDIR = bin
CXXFLAGS = -march=native -mtune=native -ftree-vectorize -O4 -Wall $(DEBUG)
CUDAFLAGS = -O4 -m64 -arch sm_21 $(DEBUG) 
CC=gcc
CXX=g++ -pthread 
MXX=mpiCC
NVCC=nvcc

# Parameter used on the reference hardware of our tests
# CUDAFLAGS = --ptxas-options=-v -O4 -m64 -arch sm_21 -Xptxas -dlcm=cg -ccbin=/usr/bin/gcc-4.5 $(DEBUG) 

OBJS := $(addprefix $(OBJDIR)/, \
	Timer.o Types.o Translator.o \
	Marshaller.o Unmarshaller.o FileWriter.o FileReader.o SocketReader.o SocketWriter.o \
	ParamHandler.o WorkloadGenerator.o EvalRunner.o SystemEvalRunner.o \
	CudaKernels.cuda.o CudaKernelsNoDup.cuda.o CudaKernelsBloom.cuda.o CudaKernelsSimpleBloom.cuda.o Main.o)

LIBS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart

TARGET = CCM

all: $(TARGET)

$(TARGET): $(OBJDIR) $(OBJS)
	 $(MXX) -o $(TARGET) $(OBJS) $(LIBS)

$(OBJDIR): 
	mkdir $(OBJDIR) 

clean:
	rm -rf $(OBJDIR) $(TARGET)

$(OBJDIR)/Timer.o : common/Timer.h common/Timer.cpp
	$(CXX) $(CXXFLAGS) -c common/Timer.cpp -o $(OBJDIR)/Timer.o

$(OBJDIR)/Translator.o : sff/simple_fwd_types.h cuda/cudaTypes.h \
	common/Translator.h common/Translator.cpp
	$(CXX) $(CXXFLAGS) -c common/Translator.cpp -o $(OBJDIR)/Translator.o

$(OBJDIR)/ParamHandler.o : eval/ParamHandler.h eval/ParamHandler.cpp
	$(CXX) $(CXXFLAGS) -c eval/ParamHandler.cpp -o $(OBJDIR)/ParamHandler.o

$(OBJDIR)/EvalRunner.o : eval/ParamHandler.h eval/WorkloadGenerator.h \
	cuda/cudaTypes.h cuda/CudaKernels.h cuda/CudaKernelsNoDup.h cuda/CudaKernelsBloom.h cuda/CudaKernelsSimpleBloom.h \
	common/Consts.h common/Timer.h common/Translator.h \
	eval/EvalRunner.h eval/EvalRunner.cpp
	$(MXX) $(CXXFLAGS) -c eval/EvalRunner.cpp -o $(OBJDIR)/EvalRunner.o

$(OBJDIR)/SystemEvalRunner.o : eval/ParamHandler.h eval/FileReader.h eval/FileWriter.h \
	cuda/cudaTypes.h cuda/CudaKernels.h cuda/CudaKernelsNoDup.h cuda/CudaKernelsBloom.h cuda/CudaKernelsSimpleBloom.h common/Timer.h \
	eval/WorkloadGenerator.h common/Translator.h \
	eval/SystemEvalRunner.h eval/SystemEvalRunner.cpp
	$(CXX) $(CXXFLAGS) -c eval/SystemEvalRunner.cpp -o $(OBJDIR)/SystemEvalRunner.o

$(OBJDIR)/WorkloadGenerator.o : common/Consts.h eval/ParamHandler.h eval/ParamHandler.cpp sff/simple_fwd_types.h \
	eval/WorkloadGenerator.h eval/WorkloadGenerator.cpp
	$(CXX) $(CXXFLAGS) -c eval/WorkloadGenerator.cpp -o $(OBJDIR)/WorkloadGenerator.o

$(OBJDIR)/Types.o : sff/siena/types.h sff/types.cc
	$(CXX) $(CXXFLAGS) -c sff/types.cc -o $(OBJDIR)/Types.o

$(OBJDIR)/CudaKernels.cuda.o : sff/siena/types.h common/Timer.h common/Consts.h \
	cuda/cudaTypes.h cuda/CudaKernels.h cuda/CudaKernels.cu
	$(NVCC) $(CUDAFLAGS) -c cuda/CudaKernels.cu -o $(OBJDIR)/CudaKernels.cuda.o

$(OBJDIR)/CudaKernelsNoDup.cuda.o : sff/siena/types.h common/Timer.h common/Consts.h \
	cuda/cudaTypes.h cuda/CudaKernelsNoDup.h cuda/CudaKernelsNoDup.cu
	$(NVCC) $(CUDAFLAGS) -c cuda/CudaKernelsNoDup.cu -o $(OBJDIR)/CudaKernelsNoDup.cuda.o

$(OBJDIR)/CudaKernelsBloom.cuda.o : sff/siena/types.h common/Timer.h common/Consts.h \
	cuda/cudaTypes.h cuda/CudaKernelsBloom.h cuda/CudaKernelsBloom.cu
	$(NVCC) $(CUDAFLAGS) -c cuda/CudaKernelsBloom.cu -o $(OBJDIR)/CudaKernelsBloom.cuda.o

$(OBJDIR)/CudaKernelsSimpleBloom.cuda.o : sff/siena/types.h common/Timer.h common/Consts.h \
	cuda/cudaTypes.h cuda/CudaKernelsSimpleBloom.h cuda/CudaKernelsSimpleBloom.cu
	$(NVCC) $(CUDAFLAGS) -c cuda/CudaKernelsSimpleBloom.cu -o $(OBJDIR)/CudaKernelsSimpleBloom.cuda.o

$(OBJDIR)/Marshaller.o: common/Consts.h cuda/cudaTypes.h \
	marshalling/Marshaller.h marshalling/Marshaller.cpp
	$(CXX) $(CXXFLAGS) -c marshalling/Marshaller.cpp -o $(OBJDIR)/Marshaller.o

$(OBJDIR)/Unmarshaller.o: common/Consts.h cuda/cudaTypes.h \
	marshalling/Unmarshaller.h marshalling/Unmarshaller.cpp
	$(CXX) $(CXXFLAGS) -c marshalling/Unmarshaller.cpp -o $(OBJDIR)/Unmarshaller.o

$(OBJDIR)/FileWriter.o : eval/WorkloadGenerator.h \
	cuda/cudaTypes.h  common/Consts.h common/Timer.h common/Translator.h \
	marshalling/Marshaller.h eval/FileWriter.h eval/FileWriter.cpp
	$(CXX) $(CXXFLAGS) -c eval/FileWriter.cpp -o $(OBJDIR)/FileWriter.o

$(OBJDIR)/SocketWriter.o : eval/WorkloadGenerator.h \
	cuda/cudaTypes.h  common/Consts.h common/Translator.h eval/ParamHandler.h \
	marshalling/Marshaller.h eval/SocketWriter.h eval/SocketWriter.cpp
	$(CXX) $(CXXFLAGS) -c eval/SocketWriter.cpp -o $(OBJDIR)/SocketWriter.o

$(OBJDIR)/FileReader.o : cuda/cudaTypes.h  common/Consts.h \
	marshalling/Unmarshaller.h eval/FileReader.h eval/FileReader.cpp
	$(CXX) $(CXXFLAGS) -c eval/FileReader.cpp -o $(OBJDIR)/FileReader.o

$(OBJDIR)/SocketReader.o : cuda/cudaTypes.h  common/Consts.h \
	marshalling/Unmarshaller.h eval/SocketReader.h eval/SocketReader.cpp
	$(CXX) $(CXXFLAGS) -c eval/SocketReader.cpp -o $(OBJDIR)/SocketReader.o

$(OBJDIR)/Main.o : common/Consts.h eval/EvalRunner.h eval/SystemEvalRunner.h eval/SocketWriter.h Main.cpp
	$(MXX) $(CXXFLAGS) -c Main.cpp -o $(OBJDIR)/Main.o
