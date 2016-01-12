# Simulation of K-means clustering with visualization

# use %>% pipe to make codes more readable
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
library(dplyr); library(tidyr)
library(ggplot2)
# patient1:100 and patient101:200 are supposed to be two clusters
# Function to generate data of j genes of 200 patients
# The strategy is to sample a value (val1) from seq(0,2, by=0.1) as mean used in rnorm()
# val1 *0.1* abs(rnorm(1, mean=0.5, sd=0.5)) is used as standard deviation to restrict
# the dispersion of the expression distribution
# for the second group, val1 and val2 are sampled from seq(14,15, by=0.1), to be different
# from the first group
generate_exprs <- function(j) {
        smlt.exprs <- matrix(nrow=200, ncol=j)
        for(i in 1:j) {
                val1 <- sample(seq(5,7, by=0.1), 1)
                val2 <- val1 *0.1* abs(rnorm(1, mean=0.5, sd = 0.5))
                rand.values <- rnorm(100, mean = val1, sd = val2)
                smlt.exprs[1:100, i] <- rand.values
                # regenerate val1 and val2 and values for the 101:200 patients
                # the sample range is changed, see how the difference can get clustered out
                val1 <- sample(seq(6,9, by=0.1), 1)
                val2 <- val1*0.1 * abs(rnorm(1, mean=0.5, sd = 0.5))
                rand.values <- rnorm(100, mean = val1, sd = val2)
                smlt.exprs[101:200, i] <- rand.values
        }
        # Assign row names "patient1:200" and col names "gene1:j"
        colnames(smlt.exprs) <- paste("gene", 1:j, sep="")
        rownames(smlt.exprs) <- paste("patient", 1:200, sep="")
        # transform to data frame
        smlt.exprs <- as.data.frame(smlt.exprs)
        return(smlt.exprs)
}


# Simulate 1 gene
# plot the data exprs vs patient 
exprs.1.gene <- generate_exprs(1)
# View the exprs
ggplot(exprs.1.gene, aes(x=c(1:200),y=as.numeric(exprs.1.gene[,1]))) + ylim(range(exprs.1.gene[,1])) + geom_point() + ggtitle("Gene exprs/patient") + 
        labs(x="Patient #", y= "exprs") 
# generate clusters by k-means
my.clusters <- kmeans(exprs.1.gene[,1], 2)
# plot again with cluster info shown by color
ggplot(exprs.1.gene, aes(x=c(1:200),y=as.numeric(exprs.1.gene[,1]), col=my.clusters$cluster)) + ylim(range(exprs.1.gene[,1])) + geom_point() + ggtitle("Gene exprs/patient") + 
        labs(x="Patient #", y= "exprs") 

# When the means sampled to generate expression values are quite different, 
# the 1 gene expression could cluster the two groups of patients pretty well. 
# However, this is very random, because the randomly sampled means could be very close. 
# When the expressions are close, the 1 gene can't be used to cluster the two groups of patients

# here 5 genes are tested
exprs.5.gene <- generate_exprs(5)
# view the data
# only expression of 1 gene is plotted
ggplot(exprs.5.gene, aes(x=c(1:200),y=as.numeric(exprs.5.gene[,1]))) + ylim(range(exprs.5.gene[,1])) +
        geom_point() + ggtitle("Gene exprs/patient") + 
        labs(x="Patient #", y= "exprs") 
# cluster
my.clusters <- kmeans(exprs.5.gene[,1:5], 2)
# plot 
ggplot(exprs.5.gene, aes(x=c(1:200),y=as.numeric(exprs.5.gene[,1]), col=my.clusters$cluster)) + 
        ylim(range(exprs.5.gene[,1])) + geom_point() + ggtitle("Gene exprs/patient") + 
        labs(x="Patient #", y= "exprs")
# with 5 genes, the two groups of patients may be well clustered even if certain gene/genes 
# has/have very close expression, which won't be distinguished individually. 
# A picture of one example is uploaded in the assignment

# Further test suggests that k-means is a good way for clustering, if certain differences
# exist between potential clusters. k-means can cluster things right, through calculating 
# the least within cluster variances. 