g++ -c main.cpp -o main.o
g++ -c hnsw.cpp -o hnsw.o
g++ main.o hnsw.o -o my_program
