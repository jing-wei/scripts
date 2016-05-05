# Using R proteomics workflow from bioconductor
# This is a run-through of the tutorial online

# Install RforProteomics and its dependencies

library("BiocInstaller")
biocLite("RforProteomics", dependencies = TRUE)

library(RforProteomics)
# Show packages
(pp<-proteomicsPackages())

# Most community-driven formats are supported by R

########
# Getting data from proteomics repositories
#########
# MS-based proteomics data is disseminated through the ProteomeXchange 
# infrastructure, which centrally coordinates submission, storage and 
# dissemination through multiple data repositories, such as the PRIDE data 
# base at the EBI for MS/MS experiments, PASSEL at the ISB for SRM data and 
# the MassIVE resource. The rpx is an interface to ProteomeXchange and 
# provides a basic access to PX data
library(rpx)

# Take one example
px <- PXDataset("PXD000001")
px
# See all files
pxfiles(px)
# Data files can be downloaded
# This takes a while
mzf <- pxget(px, pxfiles(px)[6])

######
# Handling raw MS data
#######
# mzR package provides an interface to proteowizard, to access various formats
# Data is not loaded into memory until requested
# 3 main functions
# openMSfile to create a file handle to a raw data file
# header to extract metadata about the spectra contained in the file
# peaks to extract one or multiple spectra of interest
# others: instrumentInfo, runInfo

# Here, we will work on the PXD000001 data downloaded
library("mzR")
ms <- openMSfile(mzf)
ms
str(ms)
# Use header to check the metadata of all available peaks
hd <- header(ms)
dim(hd)
# [1] 7534   21

# The variable names (columns)
colnames(hd)
str(hd)
# Header info of the 1000 
hd[1000,]
# Go back to extract data from ms, using peaks
head(peaks(ms, 1000))
# Plot it
plot(peaks(ms, 1000), type="h")

# Plot a specific slice of the raw data
# with MSmap function from MSnbase package

# Extract 1st msLevel, with retention time 30-35 min
ms1 <- which(hd$msLevel == 1)
# The retention time is in seconds. Convert to min
rtsel <- hd$retentionTime[ms1] / 60 > 30 &
        hd$retentionTime[ms1] / 60 < 35
# Map
M <- MSmap(ms, ms1[rtsel], 521, 523, .005, hd)
# Plot it
plot(M, aspect = 1, allTicks = FALSE)
# 3d 
plot3D(M)

# Add some MS spectra
i <- ms1[which(rtsel)][1]
j <- ms1[which(rtsel)][2]
M2 <- MSmap(ms, i:j, 100, 1000, 1, hd)
plot3D(M2)
