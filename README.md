# ICDE-2020-SMGCN

This is our Tensorflow implementation for the ICDE-2020 paper:
>Syndrome-aware Herb Recommendation with Multi-Graph Convolution Network

We reference the public code from https://github.com/xiangwang1223/neural_graph_collaborative_filtering

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
Jin Y, Zhang W, He X, et al. Syndrome-aware herb recommendation with multi-graph convolution network[C]//2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020: 145-156.
```
## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.8.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1


 
## Dataset
 
* `train.txt`
  * Train file.
  * Each line is a prescription split by  '\t', with the former part is symptoms split by space and the later part is herbs split by space.
  
* `test.txt`
  * Test file.
  * Each line is a prescription split by  '\t', with the former part is symptoms split by space and the later part is herbs split by space.
 
* `symPair-5.txt`
  * sym-pair file
  * Each line is a sym pair.
  
* `herbPair-40.txt`
  * herb-pair file
  * Each line is a herb pair.
 
## Example to Run the Codes
see the SMGCN.sh file  
