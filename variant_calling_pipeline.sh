##############################################################################################
# Variant calling of exome sequencing data from raw illumina reads using Mutect##############
#############################################################################################


#######
#fastqc
##########

#run this in <filePathTo>/FastQC
chmod +x fastqc
#in data folder
#job submitted to server in fastqc.sh
#output file in corresponding folders.Can be open in html format
<filePathTo>/FastQC/fastqc <filePathTo>/reads.fastq.gz

########
#fastqc provides a lot of information about the reads, including illumina versions
############
#in my case, some reads are in illumina 1.5 version, using phred64
# A good convert tool written by Heng, author of bwa
seqtk seq -VQ64 in.fq.gz > out.fq
export PATH=$PATH:<filePathTo>/seqtk


#######
#TRIMMING
################
#Trimmomatic
#http://www.usadellab.org/cms/index.php?page=trimmomatic
#Samples on two lanes are trimmed separately
#Different lanes of the same sample are to be merged after first markduplicates

#Reads are trimmed from the 3' end to have a phred score of at least 30.
#Illumina sequencing adapters are removed from the reads,
#and all reads are required to have a length of at least 36.

#Paired-end
#
java -jar <filePathTo>/Trimmomatic-0.35/trimmomatic-0.35.jar PE -phred33 <filePathTo>/reads1.fastq.gz <filePathTo>/reads2.fastq.gz -baseout <filePathTo>/trimmedReads.fastq.gz ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
#OUTPUT
#trimmedReads1P/1U/2P/2U
#Use the forward and reverse paired outputs 1P & 2P

###BWA

#MAKE INDEX FOR BWA FIRST, or download related files
#Read group info is important for GATK
#-M is required as well for compatibility

bwa mem -t4 -R '@RG\tID:<ID>\tPL:illumina\tSM:<sample>' -M <filePathTo>/human_g1k_v37.fasta.gz <filePathTo>/reads.1P.fastq.gz <filePathTo>/reads.2P.fastq.gz > <filePathTo>/mappedReads.paired.sam

##########
#picard SortSam
#########
java –jar picard.jar SortSam INPUT=unsorted.sam OUTPUT=sorted.sam SORT_ORDER=coordinate

############
#picard mark duplicates
########
#-Xmx32G depends on the server
java -Xmx32G -jar <filePathTo>/picard-tools-2.0.1/picard.jar MarkDuplicates \
INPUT=<filePathTo>/mappedReads.paired.sorted.sam \ #specify multiple times to merge
OUTPUT=<filePathTo>/mappedReads.paired.sorted.mkdup.sam \
METRICS_FILE=<filePathTo>/metrics.txt \
OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 \ #changed from default of 100
CREATE_INDEX=true \ #optional
TMP_DIR=/tmp



############
#MERGE SAMPLES ACROSS FLOWCELLS
###############
#PICARD MERGESAMFILES
java -jar picard.jar MergeSamFiles \
I=input_1.bam \
I=input_2.bam \
o=merged_files.bam

############################
#SortSam and MarkDuplicates again for merged samples, as shown above
############################


#############
#CONVERT SAM TO BAM

samtools view -bS test.sam > test.bam

#index
samtools index test.bam

###########
# indel realign
##########
# NOTE. Different versions of JAVA may be required
module load Java/1.8.0_45

java -Xmx32G -jar <filePathTo>/GenomeAnalysisTK-3.5/GenomeAnalysisTK.jar \
-T IndelRealigner -R <filePathTo>/human_g1k_v37.fasta \
-I <filePathTo>/reads.bam \
-known <filePathTo>/Mills_and_1000G_gold_standard.indels.b37.vcf \
-targetIntervals <filePathTo>/g1k_v37.realigner.intervals \
-o <filePathTo>/reads.realigned.bam

#######################
# In case the readgroup info is not right at this moment
# here's a good way to adjust for each .bam
samtools view -H $BAM | sed "s/SM:SAMPLE123/SM:SAMPLE123\tPL:ILLUMINA/g" | samtools reheader - $BAM > mybamfile.reheadered.bam

# Re index for reheadered.bam


###########
##BQSR in 4 steps
##########

# 1
#module load Java/1.8.0_45
#
#java –jar GenomeAnalysisTK.jar –T BaseRecalibrator \
#–R human.fasta \
#–I realigned.bam \
#–knownSites dbsnp137.vcf \
#–knownSites gold.standard.indels.vcf \
#–o recal.table

#2
#java –jar GenomeAnalysisTK.jar –T PrintReads \
#–R human.fasta \
#–I realigned.bam \
#–BQSR recal.table \
#–o recal.bam

#3
#java –jar GenomeAnalysisTK.jar –T BaseRecalibrator \
#–R human.fasta \
#–I realigned.bam \
#–knownSites dbsnp137.vcf \
#–knownSites gold.standard.indels.vcf \
#–BQSR recal.table \
#–o aSer_recal.table

#4
#R is required for .csv generation
module load R/3.2.1

#can keep intermediate .csv file
#java –jar GenomeAnalysisTK.jar –T AnalyzeCovariates \
#–R human.fasta \
#–before recal.table \
#–after aSer_recal.table \
#–plots recal_plots.pdf


########################
##mutect
########################

#normal vs tumor
java -Xmx32g -jar muTect-<version>.jar \
--analysis_type MuTect \
--reference_sequence Homo_sapiens_assembly19.fasta \
--dbsnp dbsnp_132_b37.leftAligned.vcf \
--cosmic hg19_cosmic_v54_120711.vcf \
--intervals <chromosome>:<region> \
#intervals are optional
--input_file:normal Normal.cleaned.bam \
--input_file:tumor Tumor.cleaned.bam \
-vcf output.vcf \
--out example.call_stats.txt \
--coverage_file example.coverage.wig.txt

#############################
#Combine vcf
java -jar GATK/3.4.0/GenomeAnalysisTK.jar \
-T CombineVariants \
-R ucsc.hg19.fasta \
-V 1.vcf \
-V 2.vcf \
-V 3.vcf \
-V etc.. \
-o output.vcf \
--filteredrecordsmergetype KEEP_IF_ANY_UNFILTERED \
--filteredAreUncalled \
--genotypemergeoption UNIQUIFY

