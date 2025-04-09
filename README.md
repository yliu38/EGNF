# Expression Graph Network Framework (EGNF)
<p align="center">
<img src="https://github.com/yliu38/EGNF/blob/main/image/overview.png" width="500">
</p>

## Environment setup
### R packages installation

**R code:**
``` r
# List of required packages
packages <- c(
  "dendextend", "tidyverse", "tibble", "gsubfn", 
  "readxl", "data.tree", "boot", "RCurl",
  "RJSONIO", "jsonlite", "foreach", "doParallel",
  "R.utils")

# Install missing packages
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load the packages
lapply(packages, library, character.only = TRUE)
```

### Python packages installation
<img src="https://github.com/yliu38/EGNF/blob/main/image/cuda_compatibility.png" width="650">

**Bash code:**
```bash
# PyTorch with GPU (please refer the image above for compatibility)
pip install torch torchvision torchaudio

# PyG dependencies (get correct versions based on your system and torch version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install torch-geometric
pip install torch-geometric

# Other packages
pip install scikit-learn scikit-optimize numpy pandas py2neo 
```

### Neo4j desktop setup
Please google neo4j desktop to download the neo4j software or you can use institutional neo4j server or neo4j clound

Below is the step-by-step instructions for using Neo4j desktop
Open the neo4j software --> click "new" --> Create project --> Add Local DBMS, input password and create --> click the project made and install Plugins of APOC and Graph Data Science Library


## Data preprocessing
The recommended input is either raw count expression matrix or normalized expression matrix like TPM. Since the network computation normally need much larger resources, **we recommend to start with matrix with around 1000 features**. 
**Some initial feature selections like differentially expressed genes (DEGs) selection are needed.**

<img src="https://github.com/yliu38/EGNF/blob/main/image/example_expression_matrix.png" width="380">

**R code:**
``` r
# load libraries
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

source("https://github.com/yliu38/EGNF/blob/main/R/functions.R")
# remove genes with 80% zeroes and na rows
exp <- remove_sparse_rows(exp)
# log2 and z-score normalization
# nor has options including "two.end", "up", "down" for choosing both high and low or high only or low only expressed clusters
exp <- norm_dat(exp, nor="down")
```

## Data split

**R code:**
``` r
set.seed(123)
n_spl = dim(exp)[2]
train_ind <- sample(1:n_spl,n_spl*0.8)
exp_train <- exp[,train_ind]
exp_test <- exp[,-train_ind]
```

## One-dimensional hierarchical clustering
### Output csv files for network construction

**R code:**
``` r
# directory is the location storing results, example can be "./folder_name/train_gene_class1_"
# group_label is your class, e.g. "primary" or "recurrent"
make_tree(exp_train_class1, directory, group_label)

# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_train_class1)
file1 <- paste0("file:/data_train/train_gene_class1_",gene_names,".csv") 

# directory is the location storing results, example can be "./folder_name/train_gene_class2_"
make_tree(exp_train_class2, directory, group_label)
# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_train_class2)
file2 <- paste0("file:/data_train/train_gene_class2_",gene_names,".csv")

url = c("URL", file1, file2)
write.table(url,"url_train.csv", sep=",",  col.names=F, row.names = F)
```
**please move the generated data_train folder and url_train.csv to the import folder of Neo4j**

## Neo4j graph network building and graph algorithm implementation
Open the neo4j software --> click the project made --> click the "..." on the right --> Open floder Import --> move the files including url_train.csv, folder for hierarchical trees to the import directory

Open terminal, run python scripts
### Build networks and implement graph-based algorithms

**Bash code:**
```python
python create_filenodes.py # creating nodes for making graph nodes
python create_nodes.py # making nodes and delete file nodes
python create_relationships.py # making edges, the default cutoff for common samples shared across edges is 4, may change according to your sample size
python output_id_table.py # output node ids for following feature selection process (id_gene_map.csv)

# after database construction, run graph algorithms including degree centrality and community detection
python project_graph_sampling_class1.py
python project_graph_sampling_class2.py # output results of algorithms, need to run this for two class separately 
```

## Feature selection--part1

**R code:**
``` r
# load graph ids
annos <- read.csv("id_gene_map.csv")

# class 1
# create matrix to store gene frequency, degree in communities
path <- "../algorithm_results/" # use the directory where you store the algorithms files starting with "Modularity_Optimization_" or "degree_cen_"
nruns <- 1e4
genes <- unique(annos$gene)
res_nw <- matrix(0,nruns,length(genes))
res_score <- matrix(0,nruns,length(genes))
colnames(res_nw) <- genes
colnames(res_score) <- genes

# fill the matrix with algorithm results
out <- matrix_out(nruns, path)
res_nw <- out$res_nw
res_score <- out$res_score

# bootstrap test (one vs all other groups)
## replace NA with 0
res_nw[is.na(res_nw)] <- 0
res_score[is.na(res_score)] <- 0

# check the distribution
summary(colSums(res_nw)); hist(colSums(res_nw))
summary(colSums(res_score)); hist(colSums(res_score))
# run bootstrap
# other p-value correction methods include "fdr", "BH", "BY"
p_table1 <- run_boot(res_nw, "bonferroni")
p_table2 <- run_boot(res_score, "bonferroni")
# do the above analysis for class 2 as well
```

## Feature selection--part2
Considering the possible unstable connection of the local machine for doing pathway enrichment, 
we recommend to run this step in terminal or server.

**Bash code:**
```bash
# the input include genes after initial selection like DEGs and files for Modularity Optimization (community detection), please revise the files accordingly
# the output is a Rdata file containing a matrix and dataframe for gene enrichment 
nohup R CMD BATCH pathway_enrich_class1.R &
nohup R CMD BATCH pathway_enrich_class2.R &
```
if you encounter "schannel: CertGetCertificateChain trust error CERT_TRUST_IS_UNTRUSTED_ROOT", please use pathway_enrich_class1_re.R and pathway_enrich_class2_re.R instead

## Feature selection--part3

**R code:**
```r
# class1
load(file="DB_pathway_class1.RData")

# a matrix to store the bootstrap result for pathway enrichment
p_table3 <- run_boot(finalMatrix, "bonferroni")
colnames(p_table3) <- c("p.value","p.adj")
rownames(p_table3) <- colnames(finalMatrix)

p_table_class1 <- cbind(p_table1, p_table2 )
colnames(p_table_class1) <- c("p.value_frequency","p.value_score","p.adj_frequency","p.adj_score")
p_table_class1$sig_or_not <- ifelse(p_table_class1$p.adj_score<0.05 & p_table_class1$p.adj_frequency<0.05, "Significant", "Not_significant")
p_table_class1$gene <- colnames(res_nw)
# processing path enrichement results
path_enrich_sub <- path_enrich[match(rownames(p_table3),path_enrich$Name),]
path_genes <- list()
for (i in seq(nrow(path_enrich_sub))){path_genes[[i]] <- path_enrich_sub$Genes[i][[1]][,2]}
all_genes <- rep(NA,length(path_genes))
for ( i in seq(length(path_genes))) {
  all_genes[i] <- paste(path_genes[[i]],collapse = "/")
}
p_table3$genes <- all_genes[match(rownames(p_table3), path_enrich_sub$Name)]
# scoring system
## include=T means including pathway enrichment filteration. include=F does not include
p_fre_sub1 <- score_gene(p_table3, p_table_class1, include=T)

# class2
load(file="DB_pathway_class2.RData")

# run the above code again
# scoring system
p_fre_sub2 <- score_gene(p_table3, p_table_class2, include=T)

# features
# only consider significant ones in terms of degree and frequency 
p_fre_sub1 <- p_fre_sub1[p_fre_sub1$sig_or_not=="Significant",]
p_fre_sub2 <- p_fre_sub2[p_fre_sub2$sig_or_not=="Significant",]
# select non-overlapping genes
tmp <- intersect(p_fre_sub1$gene,p_fre_sub2$gene)
tar <- setdiff(c(p_fre_sub1$gene,p_fre_sub2$gene), tmp)
p_fre_sub1 <- p_fre_sub1[p_fre_sub1$gene %in% tar,]
p_fre_sub2 <- p_fre_sub2[p_fre_sub2$gene %in% tar,]
# please ensure these genes exist in testing set
# select n genes for each class, here I set n=16, the recommended range is 4<n<26 
n=16
p_fre_sub1 <- p_fre_sub1[order(p_fre_sub1$sum),]; p_fre_sub2 <- p_fre_sub2[order(p_fre_sub2$sum),]
final_tar <-  c(p_fre_sub1$gene[1:n],p_fre_sub2$gene[1:n])
write.csv(final_tar, "features_unpaired.csv")
```

## Network construction for GNNs
### Clustering

**R code:**
``` r
# use the above selected features for clustering
exp_train_sub <- exp_train[final_tar,]
exp_test_sub <- exp_test[final_tar,]

# directory is the location storing results, example can be "./data_train_GNN/train_gene_"
make_tree(exp_train_sub, directory)
# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_train_sub)
file1 <- paste0("file:/data_train_GNN/train_gene_",gene_names,".csv")

# directory is the location storing results, example can be "./data_test_GNN/test_gene_"
make_tree(exp_test_sub, directory)
# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_test_sub)
file2 <- paste0("file:/data_test_GNN/test_gene_",gene_names,".csv")

url = c("URL", file1, file2)
write.table(url,"url_all.csv", sep=",",  col.names=F, row.names = F)
```
**please move the generated data_train_GNN, data_test_GNN folder and url_all.csv to the import folder of Neo4j**

### Other files needed for network construction in GNN task
A sample label file (lable_file) for the whole set, training set (lable_train) and testing set (lable_test). There are two columns, the first column for sample id and the second one is for group label.

### Build networks for running GNNs

**Bash code:**
```bash
python create_filenodes.py # creating nodes for making graph nodes, you may need to replace the url file name with yours
python create_nodes.py # making nodes and delete file nodes
python create_relationships_GNN.py # making edges

python download.network.py # output sample networks for GNNs, you may want to revise the output directory
```

## Running GNNs

**Bash code:**
```bash
# the input is files including label_file, label_train, label_test, atoms_df (sample network node), bonds_df (sample network edges)
# you may need to change number of features and output directory based on your situations.
# if you revise the input format, you would need to revise the code accordingly.
python GCN_32genes_unpaired.py
```
