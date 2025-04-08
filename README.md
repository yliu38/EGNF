# Expression graph network framework (EGNF)

## R packages installation
``` r
# List of required packages
packages <- c(
  "dendextend", "tidyverse", "tibble", "gsubfn", 
  "readxl", "data.tree"
)

# Install missing packages
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load the packages
lapply(packages, library, character.only = TRUE)
```

## Data preprocessing
### The recommended input is either raw count expression matrix or normalized expression matrix like TPM. Since the network computation normally need much larger resources, we recommend to start with matrix with around 1000 features. Some initial feature selections like differentially expressed genes (DEGs) selection are needed.
<img src="https://github.com/yliu38/EGNF/blob/main/images/example_expression_matrix.png" width="380">

``` r
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
# dat is the expression matrix; directory is the location storing generated csv files of results of hierarchical clustering for each gene, example can be "../train_gene_"
make_tree(dat, directory)

# generate url file for generating nodes in Neo4j
gene_names <- rownames(exp_train)
file <- paste0("file:/genome/glioma_train/gene_",gene_names,".csv")
url = c("URL", file)
write.table(url,"url_train.csv", sep=",",  col.names=F, row.names = F)
```

### other files needed for network construction in GNN task
A sample label file for the whole set, training set and testing set. There are two columns, the first column for sample id and the second one is for group label.

## Neo4j graph network building
