# Expression Graph Network Framework (EGNF)

If you use this tool in your work, please cite [https://doi.org/10.1101/2025.04.28.651033](https://doi.org/10.1101/2025.04.28.651033)
```bibtex
@article{liuExpressionGraphNetwork2025,
  title     = {Expression Graph Network Framework for Biomarker Discovery},
  author    = {Liu, Yang and Kannan, Kasthuri and Huse, Jason T.},
  year      = {2025},
  journal   = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.1101/2025.04.28.651033},
  url       = {https://www.biorxiv.org/content/10.1101/2025.04.28.651033v2},
  note      = {Preprint}
}
```


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [R Packages](#r-packages)
  - [Python Packages](#python-packages)
  - [Neo4j Setup](#neo4j-setup)
- [Workflow](#workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Data Split and Normalization](#2-data-split-and-normalization)  
  - [3. Hierarchical Clustering](#3-hierarchical-clustering)
  - [4. Graph Network Construction](#4-graph-network-construction)
  - [5. Graph Algorithm Implementation](#5-graph-algorithm-implementation)
  - [6. Feature Selection](#6-feature-selection)
  - [7. GNN Network Construction](#7-gnn-network-construction)
  - [8. GNN Training and Evaluation](#8-gnn-training-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Introduction

EGNF is an innovative framework for biomarker discovery designed to address complex diseases like cancer. By leveraging graph neural networks (GNNs) and advanced network-based feature engineering, EGNF uncovers predictive molecular signatures from high-dimensional gene expression data.

Key features:
- Integration of gene expression, clinical attributes, and dynamic graph representations
- Exceptional classification accuracy and interpretability
- Flexible framework applicable beyond biomedicine
- Scalable, robust, and user-friendly implementation

<p align="center">
<img src="https://github.com/yliu38/EGNF/blob/main/image/overview.png" width="550">
</p>

## Installation

### R Packages

```r
# Required packages
packages <- c(
  "dendextend", "tidyverse", "tibble", "gsubfn", 
  "readxl", "data.tree", "boot", "RCurl",
  "RJSONIO", "jsonlite", "foreach", "doParallel",
  "R.utils", "httr")

# Install missing packages
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load all packages
lapply(packages, library, character.only = TRUE)
```

### Python Packages

CUDA compatibility reference:

<p align="center">
<img src="https://github.com/yliu38/EGNF/blob/main/image/cuda_compatibility.png" width="650">
</p>

```bash
# PyTorch with GPU (select version based on your CUDA version)
pip install torch torchvision torchaudio

# PyG dependencies (adjust based on your system and torch version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install torch-geometric
pip install torch-geometric

# Other dependencies
pip install scikit-learn scikit-optimize numpy pandas py2neo argparse
```

### Neo4j Setup

1. Download and install Neo4j Desktop from the official website
2. Create a new project
3. Add a Local DBMS (set password and create)
4. Install required plugins:
   - APOC
   - Graph Data Science Library

## Workflow

### 1. Data Preprocessing

EGNF works best with raw count expression matrices or normalized expression data (e.g., TPM). To manage computational resources, we recommend starting with approximately **1,000 features**.

Initial feature selection (e.g., differentially expressed genes) is recommended before proceeding.

```r
# Load utilities
library(dendextend)
library(tidyverse)
library(tibble)
library(gsubfn)
library(readxl)
library(data.tree)
library(boot)
library(RCurl)
library(RJSONIO)
library(jsonlite)
library(foreach)
library(doParallel)
library(R.utils)
library(httr)

# Source helper functions
source("https://github.com/yliu38/EGNF/blob/main/R/functions.R")

# Remove rows with >80% zeros or NA values
exp <- remove_sparse_rows(exp)
```

### 2. Data Split and Normalization

```r
set.seed(123)
n_spl = dim(exp)[2]
train_ind <- sample(1:n_spl, n_spl*0.8)
exp_train <- exp[, train_ind]
exp_test <- exp[, -train_ind]

# Log2 transformation and z-score normalization
# Options: "two.end" (both high and low), "up" (high only), "down" (low only)
exp_train <- norm_dat(exp_train, nor="two.end")
exp_test <- norm_dat(exp_test, nor="two.end")
```

> **Note:** For paired samples or replicates, consider patient-based data splits to prevent data leakage.

### 3. Hierarchical Clustering

Generate hierarchical clustering trees and prepare files for network construction:

```r
# For Class 1 samples
# directory: location for results (e.g., "./folder_name/train_gene_class1_")
# group_label: class label (e.g., "primary" or "recurrent")
make_tree(exp_train_class1, directory, group_label)

# Generate URL file for Neo4j nodes
gene_names <- rownames(exp_train_class1)
file1 <- paste0(directory, gene_names, ".csv") 

# Repeat for Class 2 samples
make_tree(exp_train_class2, directory, group_label)
gene_names <- rownames(exp_train_class2)
file2 <- paste0(directory, gene_names, ".csv")

# Create URL file for Neo4j import
url = c("URL", file1, file2)
write.table(url, "url_train.csv", sep=",", col.names=F, row.names=F)
```

> **Important:** Move the generated data folder and url_train.csv to Neo4j's import directory.

### 4. Graph Network Construction

1. Open Neo4j Desktop
2. Access your project's import folder (via "..." menu → "Open folder Import")
3. Copy your clustering data and URL files to this folder
4. Run the following Python scripts in your terminal:

```python
# Create nodes and relationships
python create_filenodes.py     # Create file nodes
python create_nodes.py         # Convert to graph nodes and delete file nodes
python create_relationships.py  # Create edges (default cutoff: 4 shared samples)
python output_id_table.py      # Output node IDs for feature selection (id_gene_map.csv)
```

### 5. Graph Algorithm Implementation

Run graph algorithms for feature analysis:

```python
python project_graph_sampling_class1.py  # Run algorithms on Class 1
python project_graph_sampling_class2.py  # Run algorithms on Class 2
```

### 6. Feature Selection

The feature selection process involves three parts:

#### Part 1: Graph-based Features

```r
# Load graph IDs
annos <- read.csv("id_gene_map.csv")

# Set up matrices for algorithm results
path <- "../algorithm_results/"  # Directory with algorithm results
nruns <- 1e4
genes <- unique(annos$gene)
res_nw <- matrix(0, nruns, length(genes))
res_score <- matrix(0, nruns, length(genes))
colnames(res_nw) <- genes
colnames(res_score) <- genes

# Fill matrices with algorithm results
out <- matrix_out(nruns, path)
res_nw <- out$res_nw
res_score <- out$res_score

# Replace NA values with 0
res_nw[is.na(res_nw)] <- 0
res_score[is.na(res_score)] <- 0

# Check distributions
summary(colSums(res_nw))
hist(colSums(res_nw))
summary(colSums(res_score))
hist(colSums(res_score))

# Bootstrap analysis (p-value correction methods: "bonferroni", "fdr", "BH", "BY")
p_table1 <- run_boot(res_nw, "bonferroni")
p_table2 <- run_boot(res_score, "bonferroni")

# Repeat for Class 2
```

#### Part 2: Pathway Enrichment

For stability, run pathway enrichment on a server:

```bash
# Run enrichment analysis for both classes
nohup R CMD BATCH pathway_enrich_class1.R &
nohup R CMD BATCH pathway_enrich_class2.R &
```

> **Troubleshooting:** If you encounter "schannel: CertGetCertificateChain trust error CERT_TRUST_IS_UNTRUSTED_ROOT", use pathway_enrich_class1_re.R and pathway_enrich_class2_re.R instead.

#### Part 3: Feature Integration

```r
# Class 1 analysis
load(file="DB_pathway_class1.RData")

# Bootstrap for pathway enrichment
p_table3 <- run_boot(finalMatrix, "bonferroni")
colnames(p_table3) <- c("p.value", "p.adj")
rownames(p_table3) <- colnames(finalMatrix)

# Combine p-values
p_table_class1 <- cbind(p_table1, p_table2)
colnames(p_table_class1) <- c("p.value_frequency", "p.value_score", "p.adj_frequency", "p.adj_score")
p_table_class1$sig_or_not <- ifelse(p_table_class1$p.adj_score < 0.05 & 
                                   p_table_class1$p.adj_frequency < 0.05, 
                                   "Significant", "Not_significant")
p_table_class1$gene <- colnames(res_nw)

# Process pathway enrichment results
path_enrich_sub <- path_enrich[match(rownames(p_table3), path_enrich$Name),]
path_genes <- list()
for (i in seq(nrow(path_enrich_sub))) {
  path_genes[[i]] <- path_enrich_sub$Genes[i][[1]][,2]
}
all_genes <- rep(NA, length(path_genes))
for (i in seq(length(path_genes))) {
  all_genes[i] <- paste(path_genes[[i]], collapse = "/")
}
p_table3$genes <- all_genes[match(rownames(p_table3), path_enrich_sub$Name)]

# Score genes (include=T means applying pathway enrichment filter)
p_fre_sub1 <- score_gene(p_table3, p_table_class1, include=T)

# Class 2 analysis
load(file="DB_pathway_class2.RData")
# Repeat the above steps
p_fre_sub2 <- score_gene(p_table3, p_table_class2, include=T)

# Final feature selection
# Filter for significant features
p_fre_sub1 <- p_fre_sub1[p_fre_sub1$sig_or_not == "Significant",]
p_fre_sub2 <- p_fre_sub2[p_fre_sub2$sig_or_not == "Significant",]

# Select non-overlapping genes
tmp <- intersect(p_fre_sub1$gene, p_fre_sub2$gene)
tar <- setdiff(c(p_fre_sub1$gene, p_fre_sub2$gene), tmp)
p_fre_sub1 <- p_fre_sub1[p_fre_sub1$gene %in% tar,]
p_fre_sub2 <- p_fre_sub2[p_fre_sub2$gene %in% tar,]

# Select top n genes for each class (recommended: 4 < n < 26)
n <- 16
p_fre_sub1 <- p_fre_sub1[order(p_fre_sub1$sum),]
p_fre_sub2 <- p_fre_sub2[order(p_fre_sub2$sum),]
final_tar <- c(p_fre_sub1$gene[1:n], p_fre_sub2$gene[1:n])
write.csv(final_tar, "features_unpaired.csv")
```

### 7. GNN Network Construction

#### Prepare Clustering Data

```r
# Filter expression data to selected features
exp_train_sub <- exp_train[final_tar,]
exp_test_sub <- exp_test[final_tar,]

# Generate clustering trees for training data
# directory example: "./data_train_GNN/train_gene_"
make_tree(exp_train_sub, directory)
gene_names <- rownames(exp_train_sub)
file1 <- paste0("file:/data_train_GNN/train_gene_", gene_names, ".csv")

# Generate clustering trees for testing data
# directory example: "./data_test_GNN/test_gene_"
make_tree(exp_test_sub, directory)
gene_names <- rownames(exp_test_sub)
file2 <- paste0("file:/data_test_GNN/test_gene_", gene_names, ".csv")

# Create URL file
url = c("URL", file1, file2)
write.table(url, "url_all.csv", sep=",", col.names=F, row.names=F)
```

> **Important:** Move the generated data folders and url_all.csv to Neo4j's import directory.

#### Prepare Label Files

Create three label files:
1. `label_file`: Sample labels for the entire dataset
2. `label_train`: Sample labels for the training set
3. `label_test`: Sample labels for the testing set

Each file should have two columns:
- First column: Sample ID
- Second column: Group label

#### Build Sample Networks

```bash
python create_filenodes.py         # Create file nodes
python create_nodes.py             # Convert to graph nodes
python create_relationships_GNN.py  # Create edges (default cutoff: 1 shared sample)
python download.network.py         # Export sample networks for GNN
```

### 8. GNN Training and Evaluation

Run any of the GNN models with the prepared graph datasets:

```bash
# Run GCN model
python GCN_32genes_unpaired.py \
  --n_fea 32 \
  --label_file path/to/label_file.csv \
  --label_train path/to/label_train.csv \
  --label_test path/to/label_test.csv \
  --atoms_df path/to/allnodes_features_unpaired.csv \
  --bonds_df path/to/alledges_features_unpaired.csv \
  --out_dir ./results

# Run GCN-only (no edge features)
python GCN_32genes_only_unpaired.py --n_fea 32 --label_file ... (same as above)

# Run GAT model
python GAT_32genes_unpaired.py --n_fea 32 --label_file ... (same as above)

# Run GATv2 model
python GATv2_32genes_unpaired.py --n_fea 32 --label_file ... (same as above)
```

## Troubleshooting

- **Certificate errors during pathway enrichment:** Use the alternate scripts (pathway_enrich_class*_re.R)
- **Memory issues:** Reduce the number of initial features or increase system resources
- **Neo4j connection problems:** Verify Neo4j service is running and credentials are correct
- **Edge cutoff adjustments:** Modify thresholds based on your sample size in create_relationships.py

## References

1. EXPRESSION GRAPH NETWORK FRAMEWORK FOR BIOMARKER DISCOVERY, bioRxiv, https://www.biorxiv.org/content/10.1101/2025.04.28.651033v2
2. Neo4j Documentation: https://neo4j.com/docs/
3. PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
