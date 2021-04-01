import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

data_folder = 'attr/CUB_200_2011/ZSL/'
res101 = io.loadmat(data_folder + 'res101.mat')
att_splits = io.loadmat(data_folder + 'att_splits.mat')

train_loc = 'train_loc'
val_loc = 'val_loc'
trainval_loc = 'trainval_loc'
test_seen_loc = 'test_seen_loc'
test_unseen_loc = 'test_unseen_loc'
labels = res101['labels']
labels_trainval_gzsl = np.squeeze(labels[np.squeeze(att_splits[trainval_loc] - 1)])


class SJE():
    def __init__(self, feature, filtered=True, rand_seed=42, lr=0.01, margin=1, early_stop=10, epochs=100, norm_type='std'):

        self.lr = lr
        self.margin = margin
        self.early_stop = early_stop
        self.epochs = epochs
        self.norm_type = norm_type
        self.filtered = filtered

        random.seed(rand_seed)
        np.random.seed(rand_seed)

        # data_folder = '../input/xlsa17/xlsa17/xlsa17/data/CUB/'
        data_folder = 'ZSL/'
        res101 = io.loadmat(data_folder + 'res101.mat')
        att_splits = io.loadmat(data_folder + 'att_splits.mat')

        train_loc = 'train_loc'
        val_loc = 'val_loc'
        trainval_loc = 'trainval_loc'
        test_seen_loc = 'test_seen_loc'
        test_unseen_loc = 'test_unseen_loc'

        feat = feature.transpose()
        # Shape -> (dxN)
        self.X_trainval_gzsl = feat[:, np.squeeze(att_splits[trainval_loc] - 1)]
        self.X_test_seen = feat[:, np.squeeze(att_splits[test_seen_loc] - 1)]
        self.X_test_unseen = feat[:, np.squeeze(att_splits[test_unseen_loc] - 1)]

        labels = res101['labels']
        self.labels_trainval_gzsl = np.squeeze(labels[np.squeeze(att_splits[trainval_loc] - 1)])
        self.labels_test_seen = np.squeeze(labels[np.squeeze(att_splits[test_seen_loc] - 1)])
        self.labels_test_unseen = np.squeeze(labels[np.squeeze(att_splits[test_unseen_loc] - 1)])
        self.labels_test = np.concatenate((self.labels_test_seen, self.labels_test_unseen), axis=0)

        # classes that occurred in the respective set
        self.train_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[train_loc] - 1)]))
        self.val_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[val_loc] - 1)]))
        self.test_classes = np.unique(self.labels_test)

        self.trainval_classes_seen = np.unique(self.labels_trainval_gzsl)
        self.test_classes_seen = np.unique(self.labels_test_seen)
        self.test_classes_unseen = np.unique(self.labels_test_unseen)

        # split trainval into train and val according to label membership in train_classes/val_classes
        self.train_gzsl_indices = []
        self.val_gzsl_indices = []

        for cl in self.train_classes:
            self.train_gzsl_indices.extend(np.squeeze(np.where(self.labels_trainval_gzsl == cl)).tolist())

        for cl in self.val_classes:
            self.val_gzsl_indices.extend(np.squeeze(np.where(self.labels_trainval_gzsl == cl)).tolist())

        self.train_gzsl_indices = sorted(self.train_gzsl_indices)
        self.val_gzsl_indices = sorted(self.val_gzsl_indices)

        self.X_train_gzsl = self.X_trainval_gzsl[:, np.array(self.train_gzsl_indices)]
        self.labels_train_gzsl = self.labels_trainval_gzsl[np.array(self.train_gzsl_indices)]

        self.X_val_gzsl = self.X_trainval_gzsl[:, np.array(self.val_gzsl_indices)]
        self.labels_val_gzsl = self.labels_trainval_gzsl[np.array(self.val_gzsl_indices)]

        print('Tr:{}; Val:{}; Tr+Val:{}; Test Seen:{}; Test Unseen:{}\n'.format(self.X_train_gzsl.shape[1],
                                                                                self.X_val_gzsl.shape[1],
                                                                                self.X_trainval_gzsl.shape[1],
                                                                                self.X_test_seen.shape[1],
                                                                                self.X_test_unseen.shape[1]))
        # re-index labels
        i = 0
        for labels in self.trainval_classes_seen:
            self.labels_trainval_gzsl[self.labels_trainval_gzsl == labels] = i
            i += 1

        j = 0
        for labels in self.train_classes:
            self.labels_train_gzsl[self.labels_train_gzsl == labels] = j
            j += 1

        k = 0
        for labels in self.val_classes:
            self.labels_val_gzsl[self.labels_val_gzsl == labels] = k
            k += 1

        sig = att_splits['att']
        # Shape -> (Number of attributes, Number of Classes)
        self.trainval_sig = sig[:, self.trainval_classes_seen - 1]
        self.train_sig = sig[:, self.train_classes - 1]
        self.val_sig = sig[:, self.val_classes - 1]
        self.test_sig = sig[:, self.test_classes - 1]

        if self.norm_type == 'std':
            scaler_train = preprocessing.StandardScaler()
            scaler_trainval = preprocessing.StandardScaler()

            scaler_train.fit(self.X_train_gzsl.T)
            scaler_trainval.fit(self.X_trainval_gzsl.T)

            self.X_train_gzsl = scaler_train.transform(self.X_train_gzsl.T).T
            self.X_val_gzsl = scaler_train.transform(self.X_val_gzsl.T).T

            self.X_trainval_gzsl = scaler_trainval.transform(self.X_trainval_gzsl.T).T
            self.X_test_seen = scaler_trainval.transform(self.X_test_seen.T).T
            self.X_test_unseen = scaler_trainval.transform(self.X_test_unseen.T).T

    def normalizeFeature(self, x):
        # x = N x d (d:feature dimension, N:number of instances)
        x = x + 1e-10
        feature_norm = np.sum(x ** 2, axis=1) ** 0.5  # l2-norm
        feat = x / feature_norm[:, np.newaxis]

        return feat

    def find_compatible_y(self, X_n, W, y_n, sig):

        XW = np.dot(X_n, W)
        # Scale the projected vector
        XW = preprocessing.scale(XW)
        scores = np.zeros(sig.shape[1])
        scores[y_n] = 0.0
        gt_class_score = np.dot(XW, sig[:, y_n])

        for i in range(sig.shape[1]):
            if i != y_n:
                scores[i] = self.margin + np.dot(XW, sig[:, i]) - gt_class_score

        return np.argmax(scores)

    def update_W(self, X, labels, sig, W, idx):

        for j in idx:
            X_n = X[:, j]
            y_n = labels[j]
            y = self.find_compatible_y(X_n, W, y_n, sig)

            if y != y_n:
                Y = np.expand_dims(sig[:, y_n] - sig[:, y], axis=0)
                W += self.lr * np.dot(np.expand_dims(X_n, axis=1), Y)

        return W

    def zero_elements(self, W):
        W_new = np.zeros(W.shape)
        start_ids = [1, 10, 25, 40, 55, 59, 74, 80, 95, 106, 121, 136, 150, 153, 168, 183, 198, 213, 218, 223, 237, 241,
                     245, 249, 264, 279, 294, 309]
        end_ids = [9, 24, 39, 54, 58, 73, 79, 94, 105, 120, 135, 149, 152, 167, 182, 197, 212, 217, 222, 236, 240, 244,
                   248, 263, 278, 293, 308, 312]

        if W.shape[0] == 1792:
            row_start_ids = list(1 + 64 * np.arange(28))
            row_end_ids = list(64 * np.arange(1, 29))
        elif W.shape[0] == 312:
            row_start_ids = start_ids
            row_end_ids = end_ids
        else:
            print('wrong input')
        for (s, e, rs, re) in zip(start_ids, end_ids, row_start_ids, row_end_ids):
            W_new[rs - 1:re, s - 1:e] = W[rs - 1:re, s - 1:e]
        return W_new

    def fit_train(self):

        print('Training on train set...\n')

        best_val_acc = 0.0
        best_tr_acc = 0.0
        best_val_ep = -1
        best_tr_ep = -1

        rand_idx = np.arange(self.X_train_gzsl.shape[1])

        W = np.random.rand(self.X_train_gzsl.shape[0], self.train_sig.shape[0])
        W = self.normalizeFeature(W.T).T

        tr_accs = []
        val_accs = []

        for ep in range(self.epochs):

            start = time.time()

            shuffle(rand_idx)

            if self.filtered:
                W = self.zero_elements(W)
            W = self.update_W(self.X_train_gzsl, self.labels_train_gzsl, self.train_sig, W, rand_idx)

            val_acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
            tr_acc = self.zsl_acc(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)

            tr_accs.append(tr_acc)
            val_accs.append(val_acc)

            end = time.time()

            elapsed = end - start

            print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep + 1, tr_acc, val_acc,
                                                                                            elapsed // 60,
                                                                                            elapsed % 60))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_ep = ep + 1

            if tr_acc > best_tr_acc:
                best_tr_ep = ep + 1
                best_tr_acc = tr_acc

            if ep + 1 - best_val_ep > self.early_stop:
                print('Early Stopping by {} epochs. Exiting...'.format(self.epochs - (ep + 1)))
                break

        print(
            'Best Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc,
                                                                                best_tr_ep))
        print(tr_accs)
        print(val_accs)
        return W, best_val_ep

    def fit_trainval(self):

        print('\nTraining on trainval set for GZSL...\n')

        best_tr_acc = 0.0
        best_tr_ep = -1

        rand_idx = np.arange(self.X_trainval_gzsl.shape[1])

        W = np.random.rand(self.X_trainval_gzsl.shape[0], self.trainval_sig.shape[0])
        W = self.normalizeFeature(W.T).T

        tr_accs = []

        for ep in range(self.num_epochs_trainval):

            start = time.time()

            shuffle(rand_idx)

            if self.filtered:
                W = self.zero_elements(W)
            W = self.update_W(self.X_trainval_gzsl, self.labels_trainval_gzsl, self.trainval_sig, W, rand_idx)

            tr_acc = self.zsl_acc(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)

            tr_accs.append(tr_acc)

            end = time.time()

            elapsed = end - start

            print('Epoch:{}; Trainval Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep + 1, tr_acc, elapsed // 60,
                                                                                   elapsed % 60))

            if tr_acc > best_tr_acc:
                best_tr_ep = ep + 1
                best_tr_acc = tr_acc
                best_W = np.copy(W)

        print('Best Trainval Acc:{} @ Epoch {}\n'.format(best_tr_acc, best_tr_ep))
        print(tr_accs)
        return best_W

    def zsl_acc(self, X, W, y_true, sig):  # Class Averaged Top-1 Accuarcy

        XW = np.dot(X.T, W)  # N x k
        dist = 1 - spatial.distance.cdist(XW, sig.T, 'cosine')  # N x C(no. of classes)
        predicted_classes = np.array([np.argmax(output) for output in dist])
        cm = confusion_matrix(y_true, predicted_classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        acc = sum(cm.diagonal()) / sig.shape[1]

        return acc

    def zsl_acc_gzsl(self, X, W, y_true, classes, sig):  # Class Averaged Top-1 Accuarcy

        class_scores = np.matmul(np.matmul(X.T, W), sig)  # N x Number of Classes
        y_pred = np.array([np.argmax(output) + 1 for output in class_scores])

        per_class_acc = np.zeros(len(classes))

        for i in range(len(classes)):
            is_class = y_true == classes[i]
            per_class_acc[i] = ((y_pred[is_class] == y_true[is_class]).sum()) / is_class.sum()

        return per_class_acc.mean()

    def evaluate(self, zeroed=False):

        self.W_ZSL, self.num_epochs_trainval = self.fit_train()

        self.W_GZSL = self.fit_trainval()
        if zeroed:
            self.W_GZSL = self.zero_elements(self.W_GZSL)

        print('Testing...\n')

        print('generalised ZSL\n')

        acc_seen_classes = self.zsl_acc_gzsl(self.X_test_seen, self.W_GZSL, self.labels_test_seen,
                                             self.test_classes_seen,
                                             self.test_sig)
        acc_unseen_classes = self.zsl_acc_gzsl(self.X_test_unseen, self.W_GZSL, self.labels_test_unseen,
                                               self.labels_test_unseen, self.test_sig)
        HM = 2 * acc_seen_classes * acc_unseen_classes / (acc_seen_classes + acc_unseen_classes)

        print('U:{}; S:{}; H:{}'.format(acc_unseen_classes, acc_seen_classes, HM))