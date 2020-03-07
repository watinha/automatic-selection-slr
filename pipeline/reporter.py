import csv, re
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

class CSVReporter:
    def __init__ (self, filename):
        self._filename = filename
        self._precision = {}
        self._recall = {}
        self._fscore = {}
        self._exclusion_rate = {}
        self._threasholds = {}
        self._missed = {}
        self._fscore_threashold = {}
        csv_file = open(self._filename, 'w', newline='')
        self._csv_writer = csv.writer(csv_file, delimiter=';')
        self._classifiers = []
        self._run_id = 1

    def execute (self, dataset):
        keys = list(dataset.keys())
        classifiers = [ classifier_score
                for classifier_score in keys
                if re.match('.*_scores$', classifier_score) != None ]
        self._classifiers = [ re.findall('(.*)_scores$', classifier_score)[0]
                                for classifier_score in classifiers ]
        if (len(self._precision) == 0):
            self._csv_writer.writerow(['fold'] +
                    (['%s_precision' % self._classifiers[0], '%s_precision' % self._classifiers[1],
                      '%s_recall' % self._classifiers[0], '%s_recall' % self._classifiers[1],
                      '%s_fscore' % self._classifiers[0], '%s_fscore' % self._classifiers[1],
                      '%s_exclusion' % self._classifiers[0], '%s_exclusion' % self._classifiers[1],
                      '%s_threashold' % self._classifiers[0], '%s_threashold' % self._classifiers[1],
                      '%s_missed' % self._classifiers[0], '%s_missed' % self._classifiers[1],
                      '%s_fscore_th' % self._classifiers[0], '%s_fscore_th' % self._classifiers[1]]))
        for classifier_score in classifiers:
            scores = dataset[classifier_score]
            classifier_name = re.findall('(.*)_scores$', classifier_score)[0]
            precision = scores['test_precision_macro'].tolist()
            recall = scores['test_recall_macro'].tolist()
            fscore = scores['test_f1_macro'].tolist()
            exclusion_rate = scores['exclusion_rate']
            threasholds = scores['threasholds']
            missed = scores['missed']
            fscore_threashold = scores['fscore_threashold']
            if (self._precision.get(classifier_name) == None):
                self._precision[classifier_name] = []
            self._precision[classifier_name] = self._precision[classifier_name] + precision
            if (self._recall.get(classifier_name) == None):
                self._recall[classifier_name] = []
            self._recall[classifier_name] = self._recall[classifier_name] + recall
            if (self._fscore.get(classifier_name) == None):
                self._fscore[classifier_name] = []
            self._fscore[classifier_name] = self._fscore[classifier_name] + fscore
            if (self._exclusion_rate.get(classifier_name) == None):
                self._exclusion_rate[classifier_name] = []
            self._exclusion_rate[classifier_name] = self._exclusion_rate[classifier_name] + exclusion_rate
            if (self._threasholds.get(classifier_name) == None):
                self._threasholds[classifier_name] = []
            self._threasholds[classifier_name] = self._threasholds[classifier_name] + threasholds
            if (self._missed.get(classifier_name) == None):
                self._missed[classifier_name] = []
            self._missed[classifier_name] = self._missed[classifier_name] + missed
            if (self._fscore_threashold.get(classifier_name) == None):
                self._fscore_threashold[classifier_name] = []
            self._fscore_threashold[classifier_name] = self._fscore_threashold[classifier_name] + fscore_threashold

            y_score = scores['probabilities']
            y_test = scores['y_test']
            fpr, tpr, threasholds = roc_curve(y_test, y_score)
            precision, recall, threasholds2 = precision_recall_curve(y_test, y_score)
            a = auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % a)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.savefig('result/%d-roc-%s.png' % (self._run_id, classifier_name))

            plt.figure()
            lw = 2
            plt.plot(threasholds2, precision[0:len(threasholds2)], color='red',
                     lw=lw, label='Precision curve')
            plt.plot(threasholds2, recall[0:len(threasholds2)], color='blue',
                     lw=lw, label='Recall curve')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Threashold')
            plt.ylabel('Scoring')
            plt.title('Precision recall curve')
            plt.legend(loc="lower right")
            plt.savefig('result/%d-pre-rec-%s.png' % (self._run_id, classifier_name))
            #print(threasholds2)
            #print(recall)
            #print(precision)
            print('%s -> Threashold %f, Recall %f, Precision %f' %
                    (classifier_name, threasholds2[0], recall[0], precision[0]))
            print(confusion_matrix(y_test, [ 0 if i < threasholds2[0] else 1 for i in y_score ]))

        self._run_id += 1


    def report (self):
        for i in range(0, len(self._precision[self._classifiers[0]])):
            precision_values = []
            recall_values = []
            fscore_values = []
            exclusion_values = []
            threasholds_values = []
            missed_values = []
            fscore_threashold = []
            for j in self._classifiers:
                precision_values += [self._precision[j][i]]
                recall_values += [self._recall[j][i]]
                fscore_values += [self._fscore[j][i]]
                exclusion_values += [self._exclusion_rate[j][i]]
                threasholds_values += [self._threasholds[j][i]]
                missed_values += [self._missed[j][i]]
                fscore_threashold += [self._fscore_threashold[j][i]]

            row = ['Fold-%d' % i] + precision_values + recall_values + fscore_values + exclusion_values + threasholds_values + missed_values + fscore_threashold
            self._csv_writer.writerow(row)

