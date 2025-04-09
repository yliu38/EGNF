rm(list=ls())

# load libraries
library(RCurl)
library(RJSONIO)
library(jsonlite)
library(dplyr)
library(foreach)
library(doParallel)
library(R.utils)

# load exp matrix, you can any other initial feature selection other than DEGs
deseq2 <- read.csv("expression_deseq2_results.csv", row.names = 1)

#### --------------------------------------------------------------------------------------------- ####
# pathway enrichment analysis
#### --------------------------------------------------------------------------------------------- ####
DEGs <- deseq2$gene_name[1:500]

getEntrez <- postForm("https://toppgene.cchmc.org/API/lookup",
                      .opts = list(postfields = toJSON(list(Symbols = DEGs)),
                                   httpheader = c('Content-Type' = 'text/json')))
document <- fromJSON(txt=getEntrez)
geneIDs.df <- document$Genes
gene_map <- geneIDs.df

# We now run enrichment analysis
getEnrich <- postForm("https://toppgene.cchmc.org/API/enrich",
                      .opts = list(postfields = toJSON(list(Genes = gene_map$Entrez)),
                                   httpheader = c('Content-Type' = 'text/json')))
signatures <- fromJSON(txt=getEnrich)                                
all_enrich <- signatures$Annotations
path_enrich <- all_enrich %>%
  filter(QValueFDRBH < 0.05 & Category=="Pathway")
path_enrich_ini <- path_enrich
max_path <- length(unique(path_enrich_ini$Name))

# create a matrix to store the results
obtain_path <- function(x) {tryCatch(expr= {
  tmp <- postForm("https://toppgene.cchmc.org/API/enrich",
                  .opts = list(postfields = toJSON(list(Genes = x)),
                               httpheader = c('Content-Type' = 'text/json')))
  return(tmp)
},
error = function(e){
  return(obtain_path(x))
})}


cores <- detectCores()
cl <- makeCluster(cores/2) #not to overload your computer
registerDoParallel(cl)

system.time(finalMatrix <- foreach(i=1:10000, .combine=rbind, .packages=c('RCurl', 'RJSONIO', 'dplyr', 'jsonlite', 'R.utils')) %dopar% {
  dat <- read.csv(paste0("./random_genes_2pt_treatment_unpaired/post/Modularity_Optimization_",i-1,".csv"))
  tmp <- table(dat$communityId)
  tmp2 <- tmp[tmp>4]
  cid <- names(tmp2)
  # get gene list for each community
  gene_list <- list()
  for(x in seq(cid)) {
    gene_list[[x]] <- unique(dat$name[dat$communityId==cid[x]])
  }
  
  res_path <- matrix(0,1,max_path)
  colnames(res_path) <- unique(path_enrich_ini$Name)
  
  for(j in seq(gene_list)) {
    entrezIDs <- gene_map[match(gene_list[[j]], gene_map$Submitted),"Entrez"]
    entrezIDs <- na.omit(entrezIDs)
    
    # We now run enrichment analysis
    tryCatch({
      getEnrich <- withTimeout({
        obtain_path(entrezIDs)
      }, timeout = 300)
    }, TimeoutException = function(ex) {
      message("Timeout. Skipping.")
      next
    })
    
    signatures_path <- fromJSON(txt=getEnrich)                                
    all_enrich <- signatures_path$Annotations
    if (is.null(all_enrich)) {next}
    path_enrich <- all_enrich %>%
      filter(QValueFDRBH < 0.05 & Category=="Pathway")
    if(nrow(path_enrich)==0) {next}
    else {tmp3 <- res_path[1,colnames(res_path)%in%path_enrich$Name]; res_path[1,colnames(res_path)%in%path_enrich$Name] <- tmp3 + 1}
  }
  res_path
})

#stop cluster
stopCluster(cl)

save(finalMatrix, path_enrich, file = "DB_pathway_class2.RData")
