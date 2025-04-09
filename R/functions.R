# remove genes with 80% zeroes and na rows
remove_sparse_rows <- function(df, threshold = 0.8) {
  df[rowMeans(df == 0, na.rm = TRUE) < threshold & !apply(df, 1, function(x) all(is.na(x))), , drop = FALSE]
}


# log2 and z-score normalization
norm_dat <- function(df) {
  df <- log2(df+1)
  df <- apply(df,1,function(x) (x-mean(x))/sd(x))
  df <- t(df)
}


# making trees
make_tree <- function(dat, directory) {
  no_of_genes <- dim(dat)[1]
  gene_names <- rownames(dat)
  
  # Get the gene expression data
  ge_df_matrix <- as.matrix(dat)
  rownames(ge_df_matrix) <- rownames(dat)
  
  # Transpose the above matrix
  transposed_ge_df <- t(ge_df_matrix)
  transposed_ge_df <- as.data.frame(transposed_ge_df)
  
  
  # For each gene we make the tree
  for (gene_no in seq(no_of_genes)) {
    # Print the gene name and the number
    print(paste("Working on gene no.", gene_no, ":", gene_names[gene_no]))
    
    # Get the data for the gene
    gene_df <- as.data.frame(transposed_ge_df[gene_no])
    # head(gene_df)
    gene_name = colnames(gene_df)
    
    # Get the gene events
    ## Dissimilarity matrix based on euclidean distance
    d <- dist(gene_df, method = "euclidean")
    ## Hierarchical clustering using median linkage
    hc1 <- hclust(d, method = "median" )
    ## Generate dendrogram
    dend <- as.dendrogram(hc1)
    ## Get levels (depth)
    tree <- as.Node(dend)
    levels <- tree$Get("level")
    ## Return all the nodes in the tree with corresponding samples
    gene_events <- partition_leaves(dend)
    
    ## get dendrogram height
    height <- get_nodes_attr(dend, "height")
    
    
    # Create an empty data frame to append the result of each gene tree events
    df <- data.frame(matrix(ncol = 7, nrow = 0))
    
    for (i in seq(gene_events)) {
      
      # Get event characteristics
      event_no <- i
      # Get event samples
      unlisted_samples <- unlist(gene_events[i])
      samples <- paste(unlisted_samples, collapse=",")
      # Get median gene expression
      patients_ge <- gene_df %>% filter(row.names(gene_df) %in% unlisted_samples)
      median_exp <- median(patients_ge[[1]])
      
      # Get the number of samples and if there is only one sample, its a leaf
      no_of_samples <- length(unlisted_samples)
      if (no_of_samples == 1) {
        leaf_status <- 1
      } else {
        leaf_status <- 0
      }
      # Append the above data frame to df
      df <- rbind(df, c(event_no, samples, median_exp, no_of_samples, gene_name, 'gene_expression', leaf_status))
    }
    
    # add height
    df$hegith <- height
    df$levels <- levels
    # select top 10%
    df <- df[order(abs(as.numeric(df[,3])), decreasing = T),]
    nevent <- round(nrow(df) * 0.1)
    df$include <- c(rep("Y",nevent),rep("N",nrow(df)-nevent)) 
    
    # Provide column names
    colnames(df) <- c('event_no', 'samples', 'median_exp', 'no_of_samples', 'gene_name', 'molecular_data', 'leaf_status',"dendrogram_height",
                      "levels", "include")
    
    file_name = paste(directory, gene_name, ".csv", sep="")
    write.table(df, file=file_name, sep=",",row.names = F)
  }
}


# generate matrix storing gene frequency and degree
matrix_out <- function(nruns, path) {system.time(
  for(i in 1:nruns) {
    # read files and get community name
    print(paste0("the iteration ",i))
    dat <- read.csv(paste0(path, "Modularity_Optimization_",i-1,".csv"))
    degree <- read.csv(paste0(path, "degree_cen_",i-1,".csv"))
    tmp1 <- table(dat$communityId)
    # count communities with at least 5 nodes
    tmp2 <- tmp1[tmp1>4]
    dat1 <- dat[dat$communityId%in%names(tmp2),]
    dat1$spl_size <- annos[match(dat1$nodeId,annos$nodeId),"sample_size"]
    dat1$score <- degree[match(dat1$nodeId,degree$nodeId),"score"]
    dat1$score <- round(dat1$score/dat1$spl_size,3)
    
    dat1$fre <- round(1/dat1$spl_size,3)
    res_nw[i,] <- dat1[match(colnames(res_nw),dat1$name), "fre"]
    res_score[i,] <- dat1[match(colnames(res_score),dat1$name),"score"]
  })
}


# bootstrap test
## bootstrap test for means
meanfun <- function(data, i){
  d <- data[i, ]
  return(mean(d))   
}

## test procedures
run_boot <- function(dat, correction) {
  p_table <- matrix(data=NA,ncol(res_score),2)
  for(i in seq(ncol(dat))) {
    d1 <- data.frame(xs1 = dat[,i]); d2 <- data.frame(xs2 = rowMeans(dat[,-i]))
    set.seed(123)
    bo1 <- boot(d1[, "xs1", drop = FALSE], statistic=meanfun, R=1000)
    set.seed(124)
    bo2 <- boot(d2[, "xs2", drop = FALSE], statistic=meanfun, R=1000)
    res1 <- t.test(bo1$t, bo2$t, paired = FALSE, alternative = "greater", exact = TRUE)
    
    p_table[i,1] <- res1$p.value
  }
  p_table[,2] <- p.adjust(p_table[,2], method = correction, n = nrow(p_table))
  p_table <- as.data.frame(p_table)
  return(p_table)
}


# scoring system
score_gene <- function(df_path, p_fre, include = T) {
  
  scores <- function(x) {match(x, sort(unique(x)))}
  
  if (include) {
    df_long <- df_path[df_path$p.adj<0.05,c(1:4)] %>% separate_rows(genes, sep = "/"); df_long <- as.data.frame(df_long)
    p_fre$p.value_path <- df_long[match(p_fre$gene,df_long$genes),"p.adj"]
    p_fre$p.value_path[is.na(p_fre$p.value_path)] <- 1
    # filter out significant genes
    p_fre_sub <- p_fre[apply(p_fre[,c(3,4,7)], 1, function(row) any(row < 0.05)),]
    
    # Rank the values, assigning the same rank (or score) to equal values
    # Use 'ties.method = "min"' to give the same rank to tied values
    p_fre_sub$grade_fre <- scores(p_fre_sub$p.value_frequency)
    p_fre_sub$grade_score <- scores(p_fre_sub$p.value_score)
    p_fre_sub$grade_path <- scores(p_fre_sub$p.value_path)
    p_fre_sub$sum <- apply(p_fre_sub[,8:10],1,sum)
  } 
  else {
    # filter out significant genes
    p_fre_sub <- p_fre[apply(p_fre[,c(3,4)], 1, function(row) any(row < 0.05)),]
    
    # Rank the values, assigning the same rank (or score) to equal values
    # Use 'ties.method = "min"' to give the same rank to tied values
    p_fre_sub$grade_fre <- scores(p_fre_sub$p.value_frequency)
    p_fre_sub$grade_score <- scores(p_fre_sub$p.value_score)
    p_fre_sub$sum <- apply(p_fre_sub[,7:8],1,sum)
    }

  return(p_fre_sub)
}

