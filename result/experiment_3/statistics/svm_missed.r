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

first_run <- seq(3, 24, 3)
second_run <- seq(2, 24, 3)

print('--- first run ---')
print(' * missed')
print(' * svm')

target <- c(missed[first_run,'svm.k50'], missed[first_run,'svm.k100'],
            missed[first_run,'svm.k300'], missed[first_run,'svm.k1000'],
            missed[first_run,'svm.k3000'])

baseline <- c(missed_baseline[first_run,'svm.k50'], missed_baseline[first_run,'svm.k100'],
              missed_baseline[first_run,'svm.k300'], missed_baseline[first_run,'svm.k1000'],
              missed_baseline[first_run,'svm.k3000'])

shapiro.test(target)
shapiro.test(baseline)

summary(target)
summary(baseline)
wilcox.test(target, baseline, paired=T)

print('--- second run ---')
print(' * missed')
print(' * svm')

target <- c(missed[second_run,'svm.k50'], missed[second_run,'svm.k100'],
            missed[second_run,'svm.k300'], missed[second_run,'svm.k1000'],
            missed[second_run,'svm.k3000'])

baseline <- c(missed_baseline[second_run,'svm.k50'], missed_baseline[second_run,'svm.k100'],
              missed_baseline[second_run,'svm.k300'], missed_baseline[second_run,'svm.k1000'],
              missed_baseline[second_run,'svm.k3000'])

shapiro.test(target)
shapiro.test(baseline)
summary(target)
summary(baseline)
wilcox.test(target, baseline, paired=T)
