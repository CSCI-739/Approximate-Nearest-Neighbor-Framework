g++ -c -fopenmp main.cpp -o main.o
g++ -c -fopenmp hnsw_implementation/hnsw.cpp -o hnsw.o
g++ -fopenmp main.o hnsw.o -o my_program
nvcc GPU/ann.cu -o nn -O3 -arch=sm_60
