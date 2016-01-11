# Explore using Naive Bayes’ Classifiers (NBCs) to
# predict patient outcome using gene expression
# Set up function to train on n-1 patients, test on the one left out, and return the 
# predict together with actual values. 
loocv <- function(i) {
        testing_row <- i
        training_rows <- setdiff(1:nrow(tmp2), testing_row)
        training <- tmp2[training_rows,]
        testing <- tmp2[testing_row,]
        classifier <- naiveBayes(event.5 ~ ., 
                                 data = training)
        outcome.predict <- predict(classifier, testing)
        c(as.character(outcome.predict), as.character(testing$event.5))
}

# Apply loocv to each of the patients
loocv_output <- sapply(1:nrow(tmp2), loocv)

