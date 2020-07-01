library(PMCMR)

excluded <- read.table('../all-exclusion.csv', header=T, sep=',')
excluded_baseline <- read.table('../all-exclusion-baseline.csv', header=T, sep=',')
missed <- read.table('../all-missed.csv', header=T, sep=',')
missed_baseline <- read.table('../all-missed-baseline.csv', header=T, sep=',')

names(excluded) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(excluded_baseline) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(missed) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(missed_baseline) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')

classifiers <- c('dt.k50', 'dt.k100', 'dt.k300', 'dt.k1000', 'dt.k3000',
                 'svm.k50', 'svm.k100', 'svm.k300', 'svm.k1000', 'svm.k3000')

print('--- ALL ---')
print(' * excluded')

target <- c(
    excluded[,'dt.k50'], excluded[,'dt.k100'], excluded[,'dt.k300'], excluded[,'dt.k1000'], excluded[,'dt.k3000'],
    excluded[,'svm.k50'], excluded[,'svm.k100'], excluded[,'svm.k300'], excluded[,'svm.k1000'], excluded[,'svm.k3000']
)
baseline <- c(
    excluded_baseline[,'dt.k50'], excluded_baseline[,'dt.k100'], excluded_baseline[,'dt.k300'], excluded_baseline[,'dt.k1000'], excluded_baseline[,'dt.k3000'],
    excluded_baseline[,'svm.k50'], excluded_baseline[,'svm.k100'], excluded_baseline[,'svm.k300'], excluded_baseline[,'svm.k1000'], excluded_baseline[,'svm.k3000']
)

shapiro.test(target)
shapiro.test(baseline)

summary(target)
summary(baseline)

wilcox.test(target, baseline, paired=T)
