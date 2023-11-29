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

Approximate Nearest Neighbours/<br />
│<br />
├── CPU/<br />
│   ├── computations.h<br />
│   <br />
│<br />
├── GPU/<br />
│   ├── header1.h<br />
│   └── header2.h<br />
│<br />
├── hnsw_implementation/<br />
│   ├── hnsw_py.cpp<br />
│   ├── hnsw.cpp<br />
│   └── hnsw.h<br />
│<br />
├── sample_inputs/<br />
│   ├── compile_generator.sh<br />
│   └── input_generator.cc<br />
│<br />
├── sample_outputs/<br />
│   <br />
├── compile.sh<br />
│<br />
├── main.cpp<br />
│<br />
├── run.sh<br />
│<br />
└── README.md<br />
