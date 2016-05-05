
install.packages('e1071', dependencies = TRUE)
library(class)
library(e1071)

# Use the iris data set
data(iris)
# Check summary
summary(iris)
# Plot each against the others
pairs(iris[1:4], pch = 21, 
      bg = c("red", "green", "blue")[unclass(iris$Species)])

# Since the fifth column of iris indicate which species they are,
# we generate Naive Bayes Classifier using column 5 as class vector 
classifier<-naiveBayes(iris[,1:4], iris[,5])
# Compare the predicted results to the original

table(predict(classifier, iris[,1:4]), iris[,5])
# It would be more interesting if we use different data for validation