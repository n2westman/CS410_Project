#__date__ = 6/14/18
#__time__ = 4:09 PM
#__author__ = isminilourentzou

import lib
import gpustat
import numpy as np
import torch
from torchtext import data
import csv

#https://stackoverflow.com/questions/36352300/python-compute-average-of-n-th-elements-in-list-of-lists-with-different-lengths
def mean_list(a):
    return [np.mean([x[i] for x in a if len(x) > i]) for i in range(len(max(a,key=len)))]


def indices2words(indices, vocab, remove_pad=False):
    results = []
    for _ind in indices:
        words = [vocab.itos[id] for id in _ind]
        if(remove_pad): words = [word for word in words if word not in ['<pad>']]
        results.append(list(words))
    return results


def choose_selftraining_method(strategy, dataset, model, opt):
    switcher = {
        'rs': lib.active_learning.RandomSampling(dataset, opt),
        'us': lib.active_learning.UncertaintySampling(dataset, opt, model=model, method='entropy', inverse=True),
        'bald': lib.active_learning.BALD(dataset, opt, model=model, nb_MC_samples=5, inverse=True),
        'ds': lib.active_learning.DiversitySampling(dataset, opt, model=model, inverse=True),
        'st': lib.active_learning.SelfTraining(dataset, opt, model=model)
    }
    return switcher.get(strategy, lib.active_learning.RandomSampling(dataset, opt))


def choose_al_method(strategy, dataset, model, opt):
    switcher = {
        'rs': lib.active_learning.RandomSampling(dataset, opt),
        'us': lib.active_learning.UncertaintySampling(dataset, opt, model=model, method='entropy'),
        'bald': lib.active_learning.BALD(dataset, opt, model=model, nb_MC_samples=5),
        'nb': lib.active_learning.NextBatch(dataset, opt),
        'ds': lib.active_learning.DiversitySampling(dataset, opt, model=model)
    }
    return switcher.get(strategy, lib.active_learning.RandomSampling(dataset, opt))

def save_predictions(save_pred, pred, labels, tokens, binary=False):
    with open(save_pred, 'w') as file:
        if(binary):
            file.write("pred\ttarget\tsent\n")
            for prediction, label, sent in zip(pred, labels, tokens):
                file.write("{}\t{}\t{}\n".format(prediction, label, sent))
                file.write('\n')
        else:
            with open(save_pred, 'w') as file:
                file.write("pred\ttarget\tsent\n")
                for prediction, label, sent in zip(pred, labels, tokens):
                    for p, l, s in zip(prediction, label, sent):
                        if(s not in ['<pad>']):
                            file.write("{}\t{}\t{}\n".format(p, l, s))
                    file.write('\n')

def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("Gpu memory used: {}/{} \n".format(item["memory.used"], item["memory.total"]))
    return item["memory.used"]


def save_results(csv_results, averageacc, averagef1, averageprec, averagerec, averagecount, strategy):
    colnames = ['iteration', 'al', 'examples', 'acc', 'f1', 'prec', 'rec']
    with open(csv_results, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(colnames)
        for i, (count, acc, f1, prec, rec) in enumerate(zip(averagecount, averageacc, averagef1, averageprec, averagerec)):
            writer.writerow([(i+1), strategy, count, acc, f1, prec, rec])

def getState(dataset, unlabeled_entries, classifier):
    #TODO: this won't scale well - split into batches
    #TODO: make appropriate state for binary and seq.labeling
    diter = data.BucketIterator(dataset=unlabeled_entries, batch_size=len(unlabeled_entries), repeat=False, shuffle=False,
                                device=torch.device("cuda:"+str(classifier.opt.gpu) if classifier.opt.cuda else "cpu"))
    candidates = list(diter)[0]
    if isinstance(classifier, torch.nn.Module):
        _, scores, prediction = classifier(candidates)
        confidences = [o.max(0)[0].item() for o in scores]
        confidences = [pow(conf, 1. / len(y)) for conf, y in zip(confidences, prediction)]
    else:
        _, confidences, prediction = classifier(*classifier.iter_to_xy(diter))

    labeledD = None
    if(len(dataset.train.examples) > 0):
        diter = data.BucketIterator(dataset=dataset.train, batch_size=len(dataset.train), repeat=False, shuffle=False,
                                    device=torch.device("cuda:"+str(classifier.opt.gpu) if classifier.opt.cuda else "cpu"))
        labeledD = list(diter)[0]
    cand = classifier.wordrepr(candidates)
    labeledD = classifier.wordrepr(labeledD).sum(0).unsqueeze(0) if labeledD else torch.zeros((1, cand.size(1), cand.size(2)))

    labeled = [d.id for d in dataset.train.examples]
    for i, d in enumerate(unlabeled_entries.examples):
        if d.id in labeled:
            cand[i] = 0
            confidences[i] = 0
            prediction[i] = len(prediction[i])*[0]
    return cand.data.cpu().numpy(), labeledD.data.cpu().numpy(), prediction, confidences


def get_predictions(examples, fields, model, discard=False):
    if(len(examples)>0):
        pseudo_example = data.Dataset(examples=examples, fields=fields)
        pseudo_example = data.BucketIterator(dataset=pseudo_example, batch_size=model.opt.batch_size, repeat=False, shuffle=False,
                                             device=torch.device("cuda:"+str(model.opt.gpu) if model.opt.cuda else "cpu"))
        if isinstance(model, torch.nn.Module):
            golds, preds = [], [], []
            for batch in pseudo_example:
                _, _, prediction = model(batch)
                y = lib.utils.indices2words(batch.labels.data.tolist(), model.wordrepr.tag_vocab)
                golds.extend(y)
                preds.extend(prediction)
        else:
            x, golds = model.iter_to_xy(pseudo_example)
            _, _, preds = model(x, golds)

        prediction = lib.utils.indices2words(preds, model.wordrepr.tag_vocab)
        if(discard):
            temp = []
            for item in prediction:
                if(not all(i in ['O', '<pad>'] for i in item)):
                    temp.append(item)
            return temp, golds
        return prediction, golds
    return [], []