# Following tutorial by GR at http://data.princeton.edu/R/glms.html 

# Sample dataset from online
# distribution of 1607 currently married and fecund women interviewed 
# in the Fiji Fertility Survey, according to age, education, desire for 
# more children and current use of contraception
cuse <- read.table("http://data.princeton.edu/wws509/datasets/cuse.dat", header=T)

# Check data
str(cuse)
cuse
# logistic regression fit
# Contraception use depending on age, education, and desire 
# cbind is used to generate response in matrix
logit <- glm(cbind(notUsing, using) ~ age + education + wantsMore, 
             family=binomial, data=cuse)
# check the regression
plot(logit)

# Significance
1-pchisq(29.92,10)
# [1] 0.0008828339
# This indicates a better model 
# could potentially subset education, desire etc. 
# Then do regression again