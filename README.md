# Approximate-Nearest-Neighbor-Framework

# HOW TO RUN

bash compile.sh <br />
bash run.sh [input_file] [output_file] [Integer value of k] <br />

# DEMO RUN

bash compile.sh <br />
bash run.sh sample_inputs/sample4.in sample_outputs/sample4.out 5 <br />

# DEMO CONSOLE OUTPUT

For input test – Dimensions 100, Base Vectors 10000, Query Vectors 1000 <br /><br />

Total euclidean time: 11.486 sec<br />
Total HNSW time: 0.417 sec<br />
Total cosine similarity time: 11.966 sec<br />
Total cosine similarity with normalization time: 5.251 sec<br />


# FILE STRUCTURE

Approximate Nearest Neighbours/
│
├── CPU/
│   ├── computations.h
│   
│
├── GPU/
│   ├── header1.h
│   └── header2.h
│
├── hnsw_implementation/
│   ├── hnsw_py.cpp
│   ├── hnsw.cpp
│   └── hnsw.h
│
├── sample_inputs/
│   ├── compile_generator.sh
│   └── input_generator.cc
│
├── sample_outputs/
│   
├── compile.sh
│
├── main.cpp
│
├── run.sh
│
└── README.md
