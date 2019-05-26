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

################
# Handling identificatino data
#############
# The RforProteomics package distributes a small identification result 
# file (see ?TMT_Erwinia_1uLSike_Top10HCD_isol2_45stepped_60min_01.mzid) 
# that we load and parse using infrastructure from the mzID package
library("mzID")
f <- dir(system.file("extdata", package = "RforProteomics"),
         pattern = "mzid", full.names=TRUE)
#
str(f)
# chr "/Users/jingwei/Library/R/3.2/library/RforProteomics/extdata/TMT_Erwinia.mzid.gz"
basename(f)
# Read id
id <- mzID(f)
# a mzID object
# a lot of information! 
id
str(id)

# Various data can be extracted from the mzID object, using one the accessor 
# functions such as database, scans, peptides, … The object can also be 
# converted into a data.frame using the flatten function.
# The mzR package also provides support fast parsing mzIdentML files with 
# the openIDfile function

library("mzR")
# f from above
id1 <- openIDfile(f)
fid1 <- mzR::psms(id1)

head(fid1)

########
# MS/MS database search
########
# Searches are generally done without R using other softwares
# However, it can be done in R
library("rTANDEM")
?rtandem
library("shinyTANDEM")
?shinyTANDEM

fas <- pxget(px, pxfiles(px)[10])
str(fas)

# search using the MSGF+ engine
library("MSGFplus")
msgfpar <- msgfPar(database = fas,
                   instrument = 'HighRes',
                   tda = TRUE,
                   enzyme = 'Trypsin',
                   protocol = 'iTRAQ')
idres <- runMSGF(msgfpar, mzf, memory=1000)
# idres is a mzID object
# identification files
basename(mzID::files(idres)$id)
# There can be graphical user interface too
# library("MSGFgui")
# MSGFgui()

########
# Analyze search results
#######
# One starts with the construction of an MSnID object that is populated 
# with identification results that can be imported from a data.frame or 
# from mzIdenML files
library(MSnID)
# A directory holding cache files is created
msnid <- MSnID(".")
#Note, the anticipated/suggested columns in the
#peptide-to-spectrum matching results are:
#        -----------------------------------------------
#        accession
#calculatedMassToCharge
#chargeState
#experimentalMassToCharge
#isDecoy
#peptide
#spectrumFile
#spectrumID
str(msnid)
msnid <- read_mzIDs(msnid,
                    basename(mzID::files(idres)$id))
show(msnid)
# The package then enables to define, optimise and apply filtering based 
# for example on missed cleavages, identification scores, precursor mass 
# errors, etc. and assess PSM, peptide and protein FDR levels
msnid <- correct_peak_selection(msnid)
msnid$msmsScore <- -log10(msnid$`MS-GF:SpecEValue`)
msnid$absParentMassErrorPPM <- abs(mass_measurement_error(msnid))

filtObj <- MSnIDFilter(msnid)
filtObj$absParentMassErrorPPM <- list(comparison="<", threshold=5.0)
filtObj$msmsScore <- list(comparison=">", threshold=8.0)
filtObj

evaluate_filter(msnid, filtObj)
# FDR around 0.05

# Optimize filter to lower FDR
# Lower literation from 50000 to 500
filtObj.grid <- optimize_filter(filtObj, msnid, fdr.max=0.01,
                                method="Grid", level="PSM",
                                n.iter=500)
filtObj.grid