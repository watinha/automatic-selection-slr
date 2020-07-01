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
print(' * missed')

target <- c(
    missed[,'dt.k50'], missed[,'dt.k100'], missed[,'dt.k300'], missed[,'dt.k1000'], missed[,'dt.k3000'],
    missed[,'svm.k50'], missed[,'svm.k100'], missed[,'svm.k300'], missed[,'svm.k1000'], missed[,'svm.k3000']
)
baseline <- c(
    missed_baseline[,'dt.k50'], missed_baseline[,'dt.k100'], missed_baseline[,'dt.k300'], missed_baseline[,'dt.k1000'], missed_baseline[,'dt.k3000'],
    missed_baseline[,'svm.k50'], missed_baseline[,'svm.k100'], missed_baseline[,'svm.k300'], missed_baseline[,'svm.k1000'], missed_baseline[,'svm.k3000']
)

shapiro.test(target)
shapiro.test(baseline)

summary(target)
summary(baseline)

wilcox.test(target, baseline, paired=T)
