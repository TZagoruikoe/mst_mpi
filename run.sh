#!/bin/bash
make clean && make
./gen_RMAT -s 10
./gen_valid_info -in rmat-10
mpiexec -np 8 ./mst -in rmat-10
./validation -in_graph rmat-10 -in_result rmat-10.mst -in_valid rmat-10.vinfo