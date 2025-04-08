# Expression graph network framework (EGNF)

## R packages installation
``` r
# List of required packages
packages <- c(
  "dendextend", "tidyverse", "tibble", "gsubfn", 
  "readxl", "data.tree")

# Install missing packages
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load the packages
lapply(packages, library, character.only = TRUE)
```

## Neo4j desktop setup
please google neo4j desktop to download the neo4j software or you can use institutional neo4j server or neo4j clound
Open the neo4j software --> click "new" --> Create project --> Add Local DBMS, input password and create --> click the project made and install Plugins of APOC and Graph Data Science Library


## Data preprocessing
The recommended input is either raw count expression matrix or normalized expression matrix like TPM. Since the network computation normally need much larger resources, we recommend to start with matrix with around 1000 features. 
Some initial feature selections like differentially expressed genes (DEGs) selection are needed.

<img src="https://github.com/yliu38/EGNF/blob/main/images/example_expression_matrix.png" width="380">

``` r
source("https://github.com/yliu38/EGNF/blob/main/R/functions.R")
# remove genes with 80% zeroes and na rows
exp <- remove_sparse_rows(exp)
# log2 and z-score normalization
exp <- norm_dat(exp)
```

## Data split

``` r
set.seed(123)
spl = dim(exp)[2]
train_ind <- sample(1:n_spl,n_spl*0.8)
exp_train <- exp[,train_ind]
exp_test <- exp[,!colnames(exp) %in% train_ind]
```

## One-dimensional hierarchical clustering
### output csv files for network construction
For graph-based feature selection task, please use exp_train only to generate hierarchical trees for avoiding data leakage.
For building input for GNNs, please use this function for exp_train and exp_test, respectively

``` r
# dat is the expression matrix; directory is the location storing results, example can be "../train_gene_"
make_tree(dat, directory)

# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_train)
file <- paste0("file:/genome/glioma_train/gene_",gene_names,".csv")
url = c("URL", file)
write.table(url,"url_train.csv", sep=",",  col.names=F, row.names = F)
```

### other files needed for network construction in GNN task
A sample label file for the whole set, training set and testing set. There are two columns, the first column for sample id and the second one is for group label.

## Neo4j graph network building and graph algorithm implementation
Open the neo4j software --> click the project made --> click the "..." on the right --> Open floder Import --> move the files including url_train.csv, folder for hierarchical trees to the import directory

Open terminal, run python scripts
### build networks and implement graph-based algorithms
```python
python create_filenodes.py # creating nodes for making graph nodes
python create_nodes.py # making nodes and delete file nodes
python create_relationships.py # making edges

# after database construction, run graph algorithms including degree centrality and community detection
python project_graph_sampling.py # output results of algorithms 
```

### build networks for running GNNs
```python
python create_filenodes.py # creating nodes for making graph nodes
python create_nodes.py # making nodes and delete file nodes
python create_relationships_GNN.py # making edges

python download.network.py # output sample networks for GNNs
```

## R programing for obtaining selected features using algorithm results
``` r

```
