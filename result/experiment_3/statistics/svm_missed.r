library(PMCMR)

excluded <- read.table('../all-exclusion.csv', header=T, sep=',')
excluded_baseline <- read.table('../all-exclusion-baseline.csv', header=T, sep=',')
missed <- read.table('../all-missed.csv', header=T, sep=',')
missed_baseline <- read.table('../all-missed-baseline.csv', header=T, sep=',')

names(excluded) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(excluded_baseline) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(missed) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(missed_baseline) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')

dt_classifiers <- c('dt.k50', 'dt.k100', 'dt.k300', 'dt.k1000', 'dt.k3000')
svm_classifiers <- c('svm.k50', 'svm.k100', 'svm.k300', 'svm.k1000', 'svm.k3000')

print('--- first run ---')
print(' * missed')
print(' * svm')

shapiro.test(missed[,'svm.k50'])
shapiro.test(missed[,'svm.k100'])
shapiro.test(missed[,'svm.k300'])
shapiro.test(missed[,'svm.k1000'])
shapiro.test(missed[,'svm.k3000'])
shapiro.test(missed_baseline[,'svm.k50'])
shapiro.test(missed_baseline[,'svm.k100'])
shapiro.test(missed_baseline[,'svm.k300'])
shapiro.test(missed_baseline[,'svm.k1000'])
shapiro.test(missed_baseline[,'svm.k3000'])

ex_df <- missed[, svm_classifiers]
names(ex_df) <- c('target.svm.k50', 'target.svm.k100', 'target.svm.k300', 'target.svm.k1000', 'target.svm.k3000')
mat_target <- data.matrix(ex_df)
ex_df <- missed_baseline[, svm_classifiers]
names(ex_df) <- c('baseline.svm.k50', 'baseline.svm.k100', 'baseline.svm.k300', 'baseline.svm.k1000', 'baseline.svm.k3000')
mat_baseline <- data.matrix(ex_df)
mat <- cbind(mat_target, mat_baseline)

friedman.test(mat)
posthoc.friedman.nemenyi.test(mat)
posthoc.friedman.conover.test(mat)
