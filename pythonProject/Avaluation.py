def F_measure(accuracy, recall):
    top= 2 * (accuracy + recall)
    bellow= accuracy+recall
    return top/bellow
