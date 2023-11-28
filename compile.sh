g++ -c -fopenmp main.cpp -o main.o
g++ -c -fopenmp hnsw_implementation/hnsw.cpp -o hnsw.o
g++ main.o hnsw.o -o my_program
