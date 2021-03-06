# coding: utf-8

from confusion_matrix import Alphabet, ConfusionMatrix

DICT_LABEL_TO_INDEX = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4,
               'world': 5, 'finance': 6, 'game': 7, 'travel': 8, 'military': 9,
               'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
               'story': 15, 'regimen': 16, 'essay': 17}

DICT_INDEX_TO_LABEL = {index:label for label, index in DICT_LABEL_TO_INDEX.items()}


def Evaluation(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:

        gold_list = [ line.strip().split('\t')[0] for line in gold_file]
        predicted_list = [line.strip().split("\t#\t")[0] for line in predict_file]

        binary_alphabet = Alphabet()
        for i in range(18):
            binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

        cm = ConfusionMatrix(binary_alphabet)
        cm.add_list(predicted_list, gold_list)

        cm.print_out()
        macro_p, macro_r, macro_f1 = cm.get_average_prf()
        overall_accuracy = cm.get_accuracy()
        return overall_accuracy, macro_p, macro_r, macro_f1


def Evalation_lst(gold_label, predict_label, print_all=False):
    binary_alphabet = Alphabet()
    for i in range(18):
        binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

    cm = ConfusionMatrix(binary_alphabet)
    cm.add_list(predict_label, gold_label)

    if print_all:
        cm.print_out()
    overall_accuracy = cm.get_accuracy()
    return overall_accuracy

if __name__ == '__main__':
    pass