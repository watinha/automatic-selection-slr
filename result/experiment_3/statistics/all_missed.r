library(PMCMR)

excluded <- read.table('../all-exclusion.csv', header=T, sep=',')
excluded_baseline <- read.table('../all-exclusion-baseline.csv', header=T, sep=',')
missed <- read.table('../all-missed.csv', header=T, sep=',')
missed_baseline <- read.table('../all-missed-baseline.csv', header=T, sep=',')

names(excluded) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(excluded_baseline) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(missed) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
names(missed_baseline) <- c('id', 'fold', 'dt.k50', 'svm.k50', 'file', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')

classifiers <- c('dt.k50', 'svm.k50', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')
iclassifiers <- c('dt.k50', 'svm.k50', 'dt.k100', 'svm.k100', 'dt.k300', 'svm.k300', 'dt.k1000', 'svm.k1000', 'dt.k3000', 'svm.k3000')

first_run <- seq(3, 24, 3)
second_run <- seq(2, 24, 3)

print('--- first run ---')
print(' * missed')

shapiro.test(missed[first_run,'dt.k50'])
shapiro.test(missed[first_run,'dt.k100'])
shapiro.test(missed[first_run,'dt.k300'])
shapiro.test(missed[first_run,'dt.k1000'])
shapiro.test(missed[first_run,'dt.k3000'])
shapiro.test(missed[first_run,'svm.k50'])
shapiro.test(missed[first_run,'svm.k100'])
shapiro.test(missed[first_run,'svm.k300'])
shapiro.test(missed[first_run,'svm.k1000'])
shapiro.test(missed[first_run,'svm.k3000'])
shapiro.test(missed_baseline[first_run,'dt.k50'])
shapiro.test(missed_baseline[first_run,'dt.k100'])
shapiro.test(missed_baseline[first_run,'dt.k300'])
shapiro.test(missed_baseline[first_run,'dt.k1000'])
shapiro.test(missed_baseline[first_run,'dt.k3000'])
shapiro.test(missed_baseline[first_run,'svm.k50'])
shapiro.test(missed_baseline[first_run,'svm.k100'])
shapiro.test(missed_baseline[first_run,'svm.k300'])
shapiro.test(missed_baseline[first_run,'svm.k1000'])
shapiro.test(missed_baseline[first_run,'svm.k3000'])

classifiers <- c(
 'target.dt.k50', 'target.svm.k50', 'target.dt.k100', 'target.svm.k100', 'target.dt.k300',
 'target.svm.k300', 'target.dt.k1000', 'target.svm.k1000', 'target.dt.k3000', 'target.svm.k3000',
 'baseline.dt.k50', 'baseline.svm.k50', 'baseline.dt.k100', 'baseline.svm.k100', 'baseline.dt.k300',
 'baseline.svm.k300', 'baseline.dt.k1000', 'baseline.svm.k1000', 'baseline.dt.k3000', 'baseline.svm.k3000'
)

data <- data.frame(missed[first_run,iclassifiers], missed_baseline[first_run,iclassifiers])
names(data) <- classifiers
data <- data.matrix(data)
friedman.test(data)
posthoc.friedman.nemenyi.test(data)

print('--- second run ---')
print(' * missed')
print(' * dt')

shapiro.test(missed[second_run,'dt.k50'])
shapiro.test(missed[second_run,'dt.k100'])
shapiro.test(missed[second_run,'dt.k300'])
shapiro.test(missed[second_run,'dt.k1000'])
shapiro.test(missed[second_run,'dt.k3000'])
shapiro.test(missed[second_run,'svm.k50'])
shapiro.test(missed[second_run,'svm.k100'])
shapiro.test(missed[second_run,'svm.k300'])
shapiro.test(missed[second_run,'svm.k1000'])
shapiro.test(missed[second_run,'svm.k3000'])
shapiro.test(missed_baseline[second_run,'dt.k50'])
shapiro.test(missed_baseline[second_run,'dt.k100'])
shapiro.test(missed_baseline[second_run,'dt.k300'])
shapiro.test(missed_baseline[second_run,'dt.k1000'])
shapiro.test(missed_baseline[second_run,'dt.k3000'])
shapiro.test(missed_baseline[second_run,'svm.k50'])
shapiro.test(missed_baseline[second_run,'svm.k100'])
shapiro.test(missed_baseline[second_run,'svm.k300'])
shapiro.test(missed_baseline[second_run,'svm.k1000'])
shapiro.test(missed_baseline[second_run,'svm.k3000'])

classifiers <- c(
 'target.dt.k50', 'target.svm.k50', 'target.dt.k100', 'target.svm.k100', 'target.dt.k300',
 'target.svm.k300', 'target.dt.k1000', 'target.svm.k1000', 'target.dt.k3000', 'target.svm.k3000',
 'baseline.dt.k50', 'baseline.svm.k50', 'baseline.dt.k100', 'baseline.svm.k100', 'baseline.dt.k300',
 'baseline.svm.k300', 'baseline.dt.k1000', 'baseline.svm.k1000', 'baseline.dt.k3000', 'baseline.svm.k3000'
)

data <- data.frame(missed[second_run,iclassifiers], missed_baseline[second_run,iclassifiers])
names(data) <- classifiers
data <- data.matrix(data)
friedman.test(data)
posthoc.friedman.nemenyi.test(data)
