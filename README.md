# Expression graph network framework (EGNF)

## Data preprocessing
### The recommended input is either raw count expression matrix or normalized expression matrix like TPM. 
<img src="https://github.com/yliu38/EGNF/blob/main/images/example_expression_matrix.png" width="380">

``` r
# remove genes with 80% zeroes and na rows
exp <- remove_sparse_rows(exp)
# log2 and z-score normalization
exp <- norm_dat(exp)
```

##
