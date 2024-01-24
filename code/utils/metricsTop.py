import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mcsh_acc_v3,
            'MOSI1': self.__eval_mcsh_acc_2,
            'MOSEI': self.__eval_mosei_regression,
            'SIMS': self.__eval_sims_regression,
            'IEMOCAP': self.__eval_iemocap_classification,
            # 'MCSH1': self.__eval_mcsh_acc,
            # 'MCSH': self.__eval_mcsh_acc_v2,
            'MCSH': self.__eval_mcsh_acc,
        }

    def __eval_mcsh_acc(self, y_pred, y_true):

        result_string = classification_report(y_true, y_pred, digits=3, output_dict=True)
        # print(confusion_matrix(y_true, y_pred))
        # print(result_string)

        return result_string

    def __eval_mcsh_acc_2(self, y_pred, y_true):

        result_string = classification_report(y_true, y_pred, digits=3, output_dict=True)

        return result_string

    def __eval_mcsh_acc_v2(self, y_pred, y_true):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred = np.argmax(np.array(y_pred), axis=1)
        y_true = np.array(y_true)
        acc = accuracy_score(y_pred, y_true)
        F1_score = f1_score(y_true, y_pred, average='weighted')

        env_results = {
            'acc': round(acc, 4),
            'f1-score': round(F1_score, 4)
        }
        return env_results

    def __eval_mcsh_acc_v3(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_3 = np.argmax(np.array(y_pred), axis=1)
        acc = accuracy_score(y_pred_3, y_true)
        F1_score = f1_score(y_true, y_pred_3, average='weighted')

        env_results = {
            'acc': round(acc, 4),
            'f1-score': round(F1_score, 4)
        }
        return env_results

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
        # two classes
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<= 0 or > 0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        # without 0 (< 0 or > 0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        eval_results = {
            "Has0_acc_2": round(Has0_acc_2, 4),
            "Has0_F1_score": round(Has0_F1_score, 4),
            "Non0_acc_2": round(Non0_acc_2, 4),
            "Non0_F1_score": round(Non0_F1_score, 4),
            "Acc_3": round(Mult_acc_3, 4),
            "F1_score_3": round(F1_score_3, 4)
        }
        return eval_results

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true):
        exclude_zero = False
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)

        eval_results = {
            "Accuracy": accuracy_score(binary_truth, binary_preds),
            "Mult_acc_5": mult_a5,
            "Mult_acc_7": mult_a7,
            "F1 score": f_score,
            "MAE": mae,
            "Correlation Coefficient": corr,
        }
        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_preds_a2, test_truth_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr,  # Correlation Coefficient
        }
        return eval_results

    def __eval_iemocap_classification(self, y_true, y_pred):
        single = -1
        emos = ["Neutral", "Happy", "Sad", "Angry"]
        if single < 0:
            test_preds = y_pred.view(-1, 4, 2).cpu().detach().numpy()
            test_truth = y_true.view(-1, 4).cpu().detach().numpy()

            for emo_ind in range(4):
                print(f"{emos[emo_ind]}: ")
                test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
                test_truth_i = test_truth[:, emo_ind]
                f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
                acc = accuracy_score(test_truth_i, test_preds_i)
                print("  - F1 Score: ", f1)
                print("  - Accuracy: ", acc)
        else:
            test_preds = y_pred.view(-1, 2).cpu().detach().numpy()
            test_truth = y_true.view(-1).cpu().detach().numpy()

            print(f"{emos[single]}: ")
            test_preds_i = np.argmax(test_preds, axis=1)
            test_truth_i = test_truth
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
        eval_results = {
            'F1_score': f1,
            'Mult_acc_2': acc
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
