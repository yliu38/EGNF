# Expression graph network framework (EGNF)
<img src="https://github.com/yliu38/EGNF/blob/main/image/overview.png" width="750">
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
The recommended input is either raw count expression matrix or normalized expression matrix like TPM. Since the network computation normally need much larger resources, we recommend to start with matrix with around 1000 features. 
Some initial feature selections like differentially expressed genes (DEGs) selection are needed.

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
exp <- norm_dat(exp)
```

## Data split

**R code:**
``` r
set.seed(123)
spl = dim(exp)[2]
train_ind <- sample(1:n_spl,n_spl*0.8)
exp_train <- exp[,train_ind]
exp_test <- exp[,!colnames(exp) %in% train_ind]
```

## One-dimensional hierarchical clustering
### Output csv files for network construction
For graph-based feature selection task, please use exp_train only to generate hierarchical trees for avoiding data leakage.
For building input for GNNs, please use this function for exp_train and exp_test, respectively.

**R code:**
``` r
# dat is the expression matrix; directory is the location storing results, example can be "../train_gene_"
make_tree(dat, directory)

# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_train)
file <- paste0("file:/genome/glioma_train/gene_",gene_names,".csv")
url = c("URL", file)
write.table(url,"url_train.csv", sep=",",  col.names=F, row.names = F)
```

### Other files needed for network construction in GNN task
A sample label file (lable_file) for the whole set, training set (lable_train) and testing set (lable_test). There are two columns, the first column for sample id and the second one is for group label.

## Neo4j graph network building and graph algorithm implementation
Open the neo4j software --> click the project made --> click the "..." on the right --> Open floder Import --> move the files including url_train.csv, folder for hierarchical trees to the import directory

Open terminal, run python scripts
### Build networks and implement graph-based algorithms

**Bash code:**
```python
python create_filenodes.py # creating nodes for making graph nodes
python create_nodes.py # making nodes and delete file nodes
python create_relationships.py # making edges
python output_id_table.py # output node ids for following feature selection process

# after database construction, run graph algorithms including degree centrality and community detection
python project_graph_sampling.py # output results of algorithms 
```

## Feature selection--part1

**R code:**
``` r
# load graph ids
annos <- read.csv("../algorithm_results/id_gene_map_100824.csv")

# class 1
# create matrix to store gene frequency, degree in communities
path <- "../algorithm_results/random_gene_051424_pt5/algorithm_res_unpaired/"
nruns <- 1e4
genes <- unique(annos$gene)
res_nw <- matrix(0,nruns,length(genes))
res_score <- matrix(0,nruns,length(genes))
colnames(res_nw) <- genes
colnames(res_score) <- genes

# fill the matrix with algorithm results
## nruns is number of graph sampling; path is the directory storing the algorithm results
matrix_out(nruns, path)

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
# the input include genes after initial selection like DEGs and files for Modularity Optimization (community detection)
# the output is a Rdata file containing a matrix and dataframe for gene enrichment 
nohup R CMD BATCH pathway_enrich_class1.R &
nohup R CMD BATCH pathway_enrich_class2.R &
```

## Feature selection--part3

**R code:**
```r
# class1
load(file="DB_pathway_class1.RData")

# a matrix to store the bootstrap result for pathway enrichment
p_table3 <- run_boot(finalMatrix, "bonferroni")
p_table_class1 <- cbind(p_table1, p_table2 )
colnames(p_table) <- c("p.value_frequency","p.value_score","p.adj_frequency","p.adj_score")
p_table$sig_or_not <- ifelse(p_table$p.adj_score<0.05 & p_table$p.adj_frequency<0.05, "Significant", "Not_significant")
p_table$gene <- colnames(res_nw)
# processing path enrichement results
path_enrich_sub <- path_enrich[match(rownames(p_table3),path_enrich$Name),]
path_genes <- list()
for (i in seq(nrow(path_enrich_sub))){path_genes[[i]] <- path_enrich_sub$Genes[i][[1]][,2]}
all_genes <- rep(NA,length(path_genes))
for ( i in seq(length(path_genes))) {
  all_genes[i] <- paste(path_genes[[i]],collapse = "/")
}
df_path$genes <- all_genes[match(rownames(p_table3), path_enrich_sub$Name)]
# scoring system
## include=T means including pathway enrichment filteration. include=F does not include
p_fre_sub1 <- score_gene(df_path, p_fre, include=T)

# class2
load(file="DB_pathway_class2.RData")

# run the above code again
# scoring system
p_fre_sub2 <- score_gene(df_path, p_fre, include=T)

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
save(final_tar,file="unpaired_target32.Rdata")
```

## Build networks for running GNNs

**Bash code:**
```bash
python create_filenodes.py # creating nodes for making graph nodes
python create_nodes.py # making nodes and delete file nodes
python create_relationships_GNN.py # making edges

python download.network.py # output sample networks for GNNs
```

## Running GNNs

**Bash code:**
```bash
# the input is files including label_file, label_train, label_test, atoms_df (sample network node), bonds_df (sample network edges)
python GCN_32genes_unpaired.py
```
