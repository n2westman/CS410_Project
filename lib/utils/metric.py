#__date__ = 6/14/18
#__time__ = 4:08 PM
#__author__ = isminilourentzou


def eval_ner(y_true, y_pred): #(token level)
    if(len(y_pred)<=0 or len(y_true)<=0): return 0, 0, 0, 0
    pre, pre_tot, rec, rec_tot, corr, total = 0, 0, 0, 0, 0, 0
    for i in range(len(y_true)):
        y_true[i] = [word for word in y_true[i] if word not in ['<pad>']]
        y_pred[i] = y_pred[i][:len(y_true[i])]
        for j in range(len(y_true[i])):
            total += 1
            if y_pred[i][j] == y_true[i][j]:
                corr += 1
            if y_pred[i][j] not in ['O', '<pad>']:  # not 'O'
                pre_tot += 1
                if y_pred[i][j] == y_true[i][j]:
                    pre += 1
            if y_true[i][j]  not in ['O',  '<pad>']: # not 'O'
                rec_tot += 1
                if y_pred[i][j] == y_true[i][j]:
                    rec += 1
    res = corr * 1. / total if total else 0
    if pre_tot == 0: pre = 0
    else: pre = 1. * pre / pre_tot
    if rec_tot == 0: rec = 0
    else: rec = 1. * rec / rec_tot
    beta, f1score = 1, 0
    if pre != 0 or rec != 0:
        f1score = (beta * beta + 1) * pre * rec / \
                  (beta * beta * pre + rec)
    return res, f1score, pre, rec


