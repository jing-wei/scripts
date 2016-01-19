# Processed data are downloaded at http://www.ebi.ac.uk/arrayexpress/files/E-GEOD-37892/E-GEOD-37892.processed.1.zip 
# and unzipped to the working dir, with the original folder name
# Clinical information is downloaded here http://www.ebi.ac.uk/arrayexpress/files/E-GEOD-37892/E-GEOD-37892.sdrf.txt
# It's named "sample_data_relation.txt", and saved in the working directory

##################
## Read and clean up data
########################
# read clinical information from sample_data_relation file
clinical <- read.delim("sample_data_relation.txt", sep="\t", skip=8, 
                       header=T)
# Row names to source names, rm " 1" in names
# rownames(clinical) <- gsub(" 1$", "", clinical$Source.Name)
# confirm No. of stage II patients
length(which(clinical[,9] == 2))
# [1] 73

# get file names
filenms <- list.files(path="./E-GEOD-37892.processed.1/", pattern="*.txt")
# Generate patient id
patients <- gsub("_.*?.txt", "", filenms)
# Add path to filenms
filenmspath <- sapply(1:length(filenms), 
                      function(i) {paste("./E-GEOD-37892.processed.1/", filenms[i], sep="")})

# Read expression data and put into a data frame
exprs <- read.delim(filenmspath[1])
# Add the rest
for (i in 2:length(filenmspath)) {
       tmp <- read.delim(filenmspath[i])
       exprs[,i+1] <- tmp[,2]
}
# Organize exprs data
# row.names
row.names(exprs) <- exprs[,1]
# remove first colum
exprs <- exprs[,-1]
# Add patient id to exprs. Patients ids were generated afte filenms
colnames(exprs) <- patients
# remove control probes in the end
# ctl_probes <- grep("AFFX.*?", rownames(exprs))
# exprs2 <- exprs[-ctl_probes,]
# Save clinical and exprs data
write.table(clinical, file="clinical.txt", row.names=T, sep="\t")
write.table(exprs, file="exprs.txt", row.names=T, sep="\t")
# read
clinical <- read.table("clinical.txt", sep="\t")
exprs <- read.table("exprs.txt", sep="\t")

#########
# Survival test
#########
# Stage II samples
(stg2_id <- which(clinical$Characteristics.Stage. == 2))
# patient ids
stg2_patients <- (rownames(clinical)[stg2_id])
# Stage 2 exprs. without control probes
exprs_stg2 <- exprs[,stg2_patients]

# Fuzzy clustering on expression value to classify the samples into 2 classes

############

# extract stg 2 patients' clinical info: 
# id (patients)
# time (time from surgery to metastasis or last contact (if no metastasis), weeks)
# recur (metastasis or not)
# group (0 for now, later from fuzzy clustering using fanny)
clinical_stg2 <- data.frame(id=rep(0, length(stg2_id)), time=0, recur=0, group=0)
clinical_stg2$id <- gsub(" 1$", "", clinical[stg2_id, "Source.Name"])

k <- 1
for (p in stg2_patients) {
        if (!is.na(clinical[p,"Characteristics.date.at.distant.metastasis."])) {
                clinical_stg2$recur[k] <- 1
                clinical_stg2$time[k] <- round(as.numeric(difftime(clinical[p, "Characteristics.date.at.distant.metastasis."], 
                                                             clinical[p, "Characteristics.date.at.surgery."],  
                                                             units=c("weeks"))), 1)
        }
        else if (is.na(clinical[p,"Characteristics.date.at.distant.metastasis."])) {
                clinical_stg2$recur[k] <- 0
                clinical_stg2$time[k] <- round(as.numeric(difftime(clinical[p, "Characteristics.date.at.last.contact."], 
                                                             clinical[p, "Characteristics.date.at.surgery."], 
                                                             units=c("weeks"))), 1) 
        }
        k <- k + 1
}
# save clinical_stg2
write.table(clinical_stg2, file="clinical_stg2.txt", row.names=T, sep="\t")

clinical_stg2 <- read.table("clinical_stg2.txt", sep="\t")

# do survival test for each gene
# cut off p value, 0.05
library(survival)
library(cluster)
pval <- matrix(0, nrow(exprs_stg2))
# tryCatch
# takes a few minutes to finish
for (i in 1:nrow(exprs_stg2)) {
        # clustering
        tryCatch({
                # clustering
                dt <- exprs_stg2[i, ]
                fan <- fanny(t(dt), 2, memb.exp=2)
                # survival test
                cli <- clinical_stg2
                cli$group <- fan$clustering
                surv <- survdiff(Surv(time, recur) ~ group, data=cli, rho=0)
                pval[i] <- round(1 - pchisq(surv$chisq, df=1), 10)
        }, 
        warning = function(w) {
                pval[i] <- NA
        }, 
        error = function(e) {
                pval[i] <- NA
        })
}

## ?? why no NA

# save pval
write.table(pval, "pval.txt", sep="\t", row.names=F, col.names=F)

# Extract significant probes with pval<=0.05
sig_rows <- which(pval <= 0.05)
# significant genes for stage 2 samples
exprs_stg2_sig <- exprs_stg2[sig_rows, ]
# 4244 probes
# remove the last control probe
exprs_stg2_sig <- exprs_stg2_sig[-nrow(exprs_stg2_sig),]

# save
write.table(exprs_stg2_sig, "exprs_stg2_sig.txt", sep="\t", row.names=T)
exprs_stg2_sig <- read.table("exprs_stg2_sig.txt", sep="\t")

#########
# GO annotation
#######
# A-AFFY-44 - Affymetrix GeneChip Human Genome U133 Plus 2.0 [HG-U133_Plus_2]
# library platform specific db, to find entrez id ...
library("hgu133plus2.db")
library("GO.db")
# detach("package:dplyr")
# if dplyr was attached
# conflicts with select
# keytypes(hgu133plus2.db)
# match probe id to entrez id and gene names
ids <- row.names(exprs_stg2_sig)
res_go <- select(hgu133plus2.db, ids, c("ENTREZID","GO"), "PROBEID")
# save
write.table(res_go, "res_go.txt", sep="\t", row.names=T)
##########
###?? should probe ids be grouped into entrez ids?########
###########
# Use GO.db to find term associated with go.id
res_go_term <- select(GO.db, res_go$GO, "TERM", "GOID")
# Save
write.table(res_go_term, "res_go_term.txt", sep="\t", row.names=T)
# Merge probe ID, entrez ID, and GO
# retain GOID column from GO.db for later confirmation of matching
stg2_sig_go <- cbind(res_go, select(GO.db, res_go$GO, "TERM", "GOID"))
write.table(stg2_sig_go, "stg2_sig_go.txt", sep="\t", row.names=T)
# Read
stg2_sig_go <- read.table("stg2_sig_go.txt", sep="\t")

# Generate GO-term defined gene set from stg2_sig_go
# Retain gene sets with sizes > 50
library(dplyr)
# This takes about 5 mins
for (tm in stg2_sig_go$TERM) {
        if (!is.na(tm)) {
                tmp <- filter(stg2_sig_go, stg2_sig_go$TERM == tm)
                # special characterin term "/"
                tm <- gsub("/", " ", tm)
                # select by number of probes >= 50
                # if (nrow(tmp) >= 50)
                # select by number of genes >= 50
                if (length(unique(tmp$ENTREZID)) >= 50) {
                        write.table(tmp, paste("./genesets/", paste(tm, ".txt", sep=""), sep=""), 
                                    sep="\t", row.names=T)
                }
        }
}
####################
# Could select caner hallmark associated groups
# Here all groups are used, without manual selection
###########


##################
## Generate random gene sets (RGS)
###############################
# file names in ./genesets
filenms <- list.files("./genesets")
# Generate random gene sets (RGS) for each GO-term-defined geneset
# Generate ##10## RGS from each of the GO group
for (grp in filenms) {
        tmpfile <- paste("./genesets/", grp, sep="")
        tmp <- read.table(tmpfile, sep="\t")
        # 10 RGS from unique ENTREZ GENE IDs
        rgs <- replicate(10, sample(unique(tmp$ENTREZID), 30, replace=F))
        savepath <- paste("./random_genesets/", grp, sep="")
        write.table(rgs, savepath, sep = "\t", row.names=T)
}

#############
# Generate  m=36 random datasets (RDS) from stage 2 patients of GSE37892
##########
# Read previously saved clinical info
clinical_stg2 <- read.table("clinical_stg2.txt", sep="\t")
# No. of "good", non-recur
sum(!clinical_stg2$recur)
# 65
# No. of "bad"
sum(clinical_stg2$recur)
# 8
# Calculate number of good or bad tumors in RDS
# at size of about 70% of all the samples (73)
rds_recur_num <- round(0.7*sum(clinical_stg2$recur), digits=0)
# 6
(rds_nonrecur_num <- round((0.7*73 - rds_recur_num), digits=0))
# 45
####
# Generate m = 36 RDS
rds <- replicate(36, 
                 append(sample(which(clinical_stg2$recur == 1), rds_recur_num), 
                        sample(which(clinical_stg2$recur == 0), rds_nonrecur_num)))
# Save RDS
write.table(rds, "random_sample_sets.txt", sep="\t", row.names=T)
# read
rds <- read.table("random_sample_sets.txt", sep="\t")

###########
## Select the highest probe to represent the gene
##########################################
# Read the expression data with the significant probes of stage 2 tumors
exprs_stg2_sig <- read.table("exprs_stg2_sig.txt", sep="\t")
library("hgu133plus2.db")
# detach if necessary, otherwise select() won't work
detach("package:dplyr")
# Check features
keytypes(hgu133plus2.db)
probes <- row.names(exprs_stg2_sig)
gene_probe <- select(hgu133plus2.db, probes, "ENTREZID", "PROBEID")
#####QUESTION########
## Multiple entrez IDs for some probes?? more rows in gene_probes than probes
#############
# Some probes do not have gene ID, remove NA
gene_probe2 <- gene_probe[!(is.na(gene_probe$ENTREZID)),]
# Save gene probe infomation of the stage 2, significant probes
write.table(gene_probe2, "gene_probe_stg2_sig.txt", sep="\t", row.names=T)
### NOTE 
# this gene_probe has no NA in entrez id
gene_probe <- read.table("gene_probe_stg2_sig.txt", sep="\t")
# test
# any(is.na(unique(gene_probe$ENTREZID)))

length(unique(gene_probe$ENTREZID))
# For each gene ID, assign it with the maximum probe expression
library(dplyr)
# Creat matrix with unique gene ID of the significant probes in stage 2 tumors
# The colnames are the tumor samples
gene_exprs <- matrix(0, length(unique(gene_probe$ENTREZID)), ncol(exprs_stg2_sig))
colnames(gene_exprs) <- colnames(exprs_stg2_sig)
rownames(gene_exprs) <- unique(gene_probe$ENTREZID)
# it need to be converted to data frame for the loop below to work
gene_exprs <- as.data.frame(gene_exprs)
# data used ############
# gene_probe <- read.table("gene_probe_stg2_sig.txt", sep="\t")
# exprs_stg2_sig <- read.table("exprs_stg2_sig.txt", sep="\t")
###########
# For each gene in gene_exprs, select the maximum probe expression
# extract from exprs_stg2_sig
for (i in 1:nrow(gene_exprs)) {
        gen <- rownames(gene_exprs)[i]
        probeid <- as.character(gene_probe$PROBEID[which(gene_probe$ENTREZID == gen)])
        tmp <- exprs_stg2_sig[probeid,]
        maxid <- which.max(sapply(1:length(probeid), function(j) {sum(tmp[j,])}))
        maxprobe <- probeid[maxid]
        # assign
        gene_exprs[i,] <- exprs_stg2_sig[maxprobe,]
}

# save
write.table(gene_exprs, "exprs_stg2_sig_gene.txt", sep="\t", row.names=T)
# read
gene_exprs <- read.table("exprs_stg2_sig_gene.txt", sep="\t")


#############
# Screen GO-term-defined gene sets
########
m <- 36
genesets_num <- 10

# screen for each GO group
# RGS in ./random_genesets # grouped by GO TERM
# RDS in ./random_sample.sets.txt
# Clinical info in ./clinical_stg2.txt
# Gene expression data in ./exprs_stg2_sig_gene.txt
rds <- read.table("random_sample_sets.txt", sep="\t")
clinical <- read.table("clinical_stg2.txt", sep="\t")
exprs <- read.table("exprs_stg2_sig_gene.txt", sep="\t")
# palce to hold p values
pval <- matrix(0, genesets_num, m)
## a for loop is used to go through each GO group
## practically, this could be reduced, if some GO groups are manually selected
rgsfile <- list.files("./random_genesets")
rgsfilepath <- paste("./random_genesets/", rgsfile, sep="")
library(cluster)
library(survival)
for (rfp in rgsfilepath) {
        rgs <- read.table(rfp, sep="\t")
        for (i in 1:m) {
                for (j in 1:genesets_num) {
                        tryCatch({
                                dt <- exprs[rgs[,j], rds[,i]]
                                fan <- fanny(t(dt), 2, memb.exp=2)
                                # survival
                                dt_cli <- clinical[rds[,i], ]
                                dt_cli$group <- fan$clustering
                                surv <- survdiff(Surv(time, recur) ~ group, data=dt_cli, rho=0)
                                if (round(1 - pchisq(surv$chisq, df=1), 10) <= 0.05) {
                                        pval[j, i] <- 1
                                } 
                                else if (round(1 - pchisq(surv$chisq, df=1), 10) > 0.05) {
                                        pval[j, i] <- 0
                                }
                        }, 
                        warning = function(w) {
                                pval[j, i] <- NA
                        }, 
                        error = function(e) {
                                pval[j, i] <- NA
                        })
                }
        }
        # write for each GO group
        pval_filepath <- gsub(".txt$", "_pval.txt", rfp)
        write.table(pval, pval_filepath, sep="\t", row.names=T)
}

##########
## QUESTION
###########     
# why doesn't it work without tryCatch
# Error in survdiff.fit(y, groups, strata.keep, rho) : 
# There is only 1 group



###########
## Rank
##########

# for each RGS, calculate the fraction of RDSs for which it is predictive (%)
rgs_pval_file <- list.files("./random_genesets")[grep(".*?_pval.*?", list.files("./random_genesets"))]
rgs_pval_filepath <- paste("./random_genesets/", rgs_pval_file, sep="")
for (pv in rgs_pval_filepath) {
        rgs_pval <- read.table(pv, sep="\t")
        pred_rate <- sapply(1:nrow(rgs_pval), function(s) {round(sum(rgs_pval[s])/ncol(rgs_pval), digits=2)})
        rgs_pval$pred <- pred_rate
        pred_filepath <- gsub(".txt$", "_pred.txt", pv)
        write.table(rgs_pval, pred_filepath, sep="\t", row.names=T)
}

# Extract RGSs which are predictive > 0.9
# Rows with pred > 0.9 in *_pval_pred.txt files correspond to columns in 
# RGS file 
rgs_pred_file <- list.files("./random_genesets")[grep(".*?_pred.*?", list.files("./random_genesets"))]
rgs_pred_filepath <- paste("./random_genesets/", rgs_pred_file, sep="")
pred_cutoff <- 0.9
for (pred in rgs_pred_filepath) {
        rgs_pred <- read.table(pred, sep="\t")
        # rows
        pred_sig_id <- which(rgs_pred$pred >= pred_cutoff)
        # Read rgs data
        rgsfilepath <- gsub("_pval_pred.txt", ".txt", pred)
        rgs <- read.table(rgsfilepath, sep="\t")
        # select columns in rgs
        sig_rgs <- rgs[ ,pred_sig_id]
        # write file
        sig_rgsfilepath <- gsub(".txt", "_sig.txt", rgsfilepath)
        write.table(sig_rgs, sig_rgsfilepath, sep="\t", row.names=T)
}

########
## Top genes
#######
# No RGS predictive over 0.9 is returned. The _sig.txt files are empty. 
# This is probably due to the extremely low random gene sets generated (10)
# If any RGS were returned significant, the frequency of each unique gene 
# can be counted. The top 30 genes by frequency can be selected. 




#######
## Rerun to generate another top 30
########