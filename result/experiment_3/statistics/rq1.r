library(PMCMR)

fscore <- read.table('../all-fscore.csv', header=T, sep=',')
precision <- read.table('../all-precision.csv', header=T, sep=',')
recall <- read.table('../all-recall.csv', header=T, sep=',')
names(fscore) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(precision) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(recall) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')

classifiers <- c('dt.k50', 'svm.k50', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')

print('--- FSCORE ---')
for (i in classifiers) {
    r <- shapiro.test(fscore[, i])
    print(paste(' - Classifier: ', i))
    print(r)
}

mat <- data.matrix(fscore[, classifiers])
friedman.test(mat)
posthoc.friedman.nemenyi.test(mat)

print('--- Precision ---')
for (i in classifiers) {
    r <- shapiro.test(precision[, i])
    print(paste(' - Classifier: ', i))
    print(r)
}

mat <- data.matrix(precision[, classifiers])
friedman.test(mat)
posthoc.friedman.nemenyi.test(mat)

print('--- Recall ---')
for (i in classifiers) {
    r <- shapiro.test(recall[, i])
    print(paste(' - Classifier: ', i))
    print(r)
}

mat <- data.matrix(recall[, classifiers])
friedman.test(mat)
posthoc.friedman.nemenyi.test(mat)
