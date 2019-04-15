

# This is to create protein databases containing uniprotKB or swissprot, and corresponding
# dbSNP/Cosmic missense info from homo_sapiens_variation.txt (downloaded from uniprot)

# For each missense mutation found for each protein, peptides (containing 14 AAs flanking mutation site if possible)
# will be appended to the end of corresponding protein, separated by "X"



# 
study_dir <- "/Users/[USER]/R/database" # adjust accordingly
setwd(study_dir)

library(dplyr)
library(stringr)
library(data.table)
library(seqinr)
library(doParallel)
# use a() in seqinr package to map three letter AA code to one letter code



#######################
# swissprot
#######################

# put swissprot (can+iso) into hash table
swissprot <- read.fasta(file="swissprot_can_iso.fasta")

# two hash environment, One to update, one for reference
swissenv_export <- new.env(hash=TRUE)
swissenv <- new.env(hash=TRUE)

# Acession only
swissprot_ac <- vector()
for(i in seq_along(swissprot)) {
  swissprot_ac[i] <- make.names(str_split(names(swissprot)[i], "\\|")[[1]][2])
}
# hash table for updating; collapsed
for(i in seq_along(swissprot)) {
  tmp_ac <- swissprot_ac[i]
  swissenv_export[[tmp_ac]] <- paste0(as.character(swissprot[[i]]), collapse="")
}


# A second hash swissprot info in environment as reference to generate mutant peptides
for(i in seq_along(swissprot)) {
  tmp_ac <- swissprot_ac[i]
  swissenv[[tmp_ac]] <- as.character(swissprot[[i]])
}


# read variant in line by line
# The first row is only lines...
var_data <- data.table::fread(input="homo_sapiens_variation.txt", skip=145L)


####### Utility functions

# get position & seq information of variant and corresponding protein AC
# also clinical info if any
getVarInfo <- function(x) {
  require(seqinr)
  require(dplyr)
  # location of mutant
  loc <- regmatches(var_data[x, "Variant AA Change"], regexpr("[[:digit:]]+", var_data[x, "Variant AA Change"]))
  # mutant AA in single amino acid code
  mut <- a(substr(var_data[x, "Variant AA Change"], nchar(var_data[x, "Variant AA Change"])-2, nchar(var_data[x, "Variant AA Change"]))) %>% tolower()
  # get protein AC
  prot_ac <- make.names(var_data[x, "AC"])
  # consequence type
  cons_type <- var_data[x, "Consequence Type"]
  # create header
  #header <- paste0(var_data[x, "AC"], "_", var_data[x, "Variant AA Change"], "_", var_data[x, "Source DB ID"])
  header <- sprintf("%s_%s_%s", var_data[x, "AC"], var_data[x, "Variant AA Change"], var_data[x, "Source DB ID"])
  # clinical info
  #clinical <- paste0(var_data[x, 6:8], collapse=";")
  clinical <- sprintf("%s;%s;%s", var_data[x, 6], var_data[x,7], var_data[x,8])
  #return (loc); return (mut); return (prot_ac)
  return (list(loc=loc, mut=mut, prot_ac=prot_ac, cons_type=cons_type, header=header, clinical=clinical))
}

# 2
getVarInfo2 <- function(x) {
        require(seqinr)
        require(dplyr)
        curr_row <- var_data[x,]
        # location of mutant
        loc <- regmatches(curr_row[["Variant AA Change"]], regexpr("[[:digit:]]+", curr_row[["Variant AA Change"]]))
        # mutant AA in single amino acid code
        mut <- a(substr(curr_row[["Variant AA Change"]], nchar(curr_row[["Variant AA Change"]])-2, nchar(curr_row[["Variant AA Change"]]))) %>% tolower()
        # get protein AC
        prot_ac <- make.names(curr_row[["AC"]])
        # consequence type
        cons_type <- curr_row[["Consequence Type"]]
        # create header
        #header <- paste0(var_data[x, "AC"], "_", var_data[x, "Variant AA Change"], "_", var_data[x, "Source DB ID"])
        header <- sprintf("%s_%s_%s", curr_row[["AC"]], curr_row[["Variant AA Change"]], curr_row[["Source DB ID"]])
        # clinical info
        #clinical <- paste0(var_data[x, 6:8], collapse=";")
        clinical <- sprintf("%s;%s;%s", curr_row[[6]], curr_row[[7]], curr_row[[8]])
        #return (loc); return (mut); return (prot_ac)
        return (list(loc=loc, mut=mut, prot_ac=prot_ac, cons_type=cons_type, header=header, clinical=clinical))
}


# get left or right seq
getLeft <- function(loc, prot_ac) {
  leftSeq <- vector()
  prot_seq <- swissenv[[prot_ac]]
  if(! is.null(prot_seq)) {
    if (loc < 15) {leftSeq <- paste0(prot_seq[1:(loc-1)], collapse="")} 
    else if (loc >=15) {leftSeq <- paste0(prot_seq[(loc-14):(loc-1)], collapse="")}
  } else {
    leftSeq <- NULL
  }
  return(leftSeq)
}

getRight <- function(loc, prot_ac) {
  rightSeq <- vector()
  prot_seq <- swissenv[[prot_ac]]
  len <- length(prot_seq)
  if(! is.null(prot_seq)) {
    if ((loc+14) > len) {rightSeq <- paste0(prot_seq[(loc+1):len], collapse="")}
    else if ((loc+14) <= len) {rightSeq <- paste0(prot_seq[(loc+1):(loc+14)], collapse="")}
  } else {
    rightSeq <- NULL
  }
  return(rightSeq)
}


# with info from getVarInfo, get mut sequence
getMutPep <- function(loc, mut, prot_ac) {
  mutPep <- vector()
  mutPep <- sprintf("%s%s%s", getLeft(loc, prot_ac), mut, getRight(loc, prot_ac))
  if (nchar(mutPep) ==1) {mutPep <- NULL}
  return(mutPep)
}

###############################################################
# loop for each row of var_data, skip first row
## do parallel
#cl <- makeCluster(3)  
#registerDoParallel(cl)
############
options(warn=-1)

mutPep <- list()
mutHeader <- list()
k <- 1
for(i in seq(from=2, nrow(var_data))) {
  require(seqinr)
  varInfo <- getVarInfo(i)
  loc <- as.numeric(varInfo$loc)
  mut <- varInfo$mut
  prot_ac <- varInfo$prot_ac
  cons_type <- varInfo$cons_type
  if(! is.na(mut) & !is.null(swissenv[[prot_ac]]) & grepl("missense", cons_type)) {
    mutPepTmp <- getMutPep(loc, mut, prot_ac)
    # update hashed key value
    
    #swissenv_export[[prot_ac]] <- paste0(append(swissenv_export[[prot_ac]], c("x", mutPepTmp)), collapse="")
    swissenv_export[[prot_ac]] <- sprintf("%s%s%s", swissenv_export[[prot_ac]], "x", mutPepTmp)
    # save variant seq and header
    #if (! is.null(mutPepTmp)) {
      mutPep[[k]] <- mutPepTmp
      #mutHeader[[k]] <- paste0(varInfo$header, " ", varInfo$clinical, collapse="")
      mutHeader[[k]] <- sprintf("%s %s", varInfo$header, varInfo$clinical)
      k <- k+1
  #}
  }
}

options(warn=0)

# write variant only
write.fasta(sequences=mutPep, names=mutHeader, 
            file.out="/Users/[USER]/R/database/varSeqDB_swissprot.fasta", open="w", nbchar=60, as.string=FALSE)



### Extract names including info in fasta headers
### FULL names
swissprot_nms <- vector()
for(i in seq_along(swissprot)) {
        swissprot_nms[i] <- attr(swissprot[[i]], "Annot")
}

# extract appened DB from env
# for (item in ls(swissenv)) assign(item get(item, swissenv))
swissVarHeader <- list()
swissVarSeq <- list()
q <- 1
for (fullName in swissprot_nms) {
  ac <- make.names(str_split(fullName, "\\|")[[1]][2])
  swissVarHeader[[q]] <- fullName
  swissVarSeq[[q]] <- swissenv_export[[ac]]
  q <- q+1
}
# save
write.fasta(sequences=swissVarSeq, names=swissVarHeader, 
            file.out="/Users/[USER]/R/database/swissprotVar.fasta", open="w", nbchar=60, as.string=FALSE)

# for timing
### Start the clock!
##ptm <- proc.time()
##
##
### Stop the clock
##proc.time() - ptm
##