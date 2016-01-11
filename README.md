# Bayesian partition method to learn co-modules in two matrices
This repository contains a statistical model to find _co-module_ in two matrices
sharing a common dimension. For details of the model please see http://bioinformatics.oxfordjournals.org/content/28/7/955.abstract.

## License
This software is distributed under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

# Introduction

Suppose we have two matrices _Y_ and _Z_ with
dimension _d_ by _g_ and _p_ by _g_ respectively. A co-module is a pair of sub-matrices
in _Y_ and _Z_ that share a significant pattern in contrast to other values. It is 
related to bi-cluster problems that divide a single matrix into non-overlapping
blocks that values within a block share similar pattern. Co-module extends this concept to 
two matrices. 

# About the folders
* _code_ contains the source code of implementing the model
* _realData_ contains the network data used in the original paper

## About real data
The real data used in the published paper contain the drug-target 
relations downloaded from [DrugBank database](http://www.drugbank.ca), disease-gene 
relations downloaded from [OMIM database](http://www.omim.org) and 
a comprehensive protein-protein iteration network generated from multiple reliable
sources (see details in the [paper](http://bioinformatics.oxfordjournals.org/content/28/7/955.abstract)).
If the real data are used in your publication please notify the author and cite the paper.

# Code

## Compilation
The code is written in C. Use Makefile to compile the source code to an executable file. 
It requires installation of GNU Scientific Library (GSL). 
Please make sure the library has been installed in your OS and the path is correctly specified 
in _Makefile_.

## Running
The excutable file will read _configure_ in the folder before running. 
The input files and parameters are needed to specified in the configure file.
