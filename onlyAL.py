# -*- coding: utf-8 -*-
#__date__ = 7/26/18
#__time__ = 10:51 AM
#__author__ = isminilourentzou


import logging
from itertools import count
import os
import numpy as np
from config import opt, additional_args
import lib
import time
start_time = time.time()
additional_args(opt)
N_SAMPLES = opt.nsamples

logging.basicConfig(filename=os.path.join(opt.save_dir, opt.al_method+'only.log') if opt.logfolder else None, level=logging.INFO)
logging.getLogger("data").setLevel('WARNING')
logging.getLogger("model").setLevel('WARNING')
logging.getLogger("train").setLevel('WARNING')
logger = logging.getLogger("onlyAL")

def main():
    logger.info(opt)
    # load the test data: target languages
    dataset = lib.data.Conll_dataset(opt, tag_type='ner', train=False)
    #TODO: Sample initial train and validation set?
    #dataset.sample_validation(100)
    wordrepr =  lib.train.build_wordrepr(opt, dataset.vocabs)
    logger.info('Begin training active learning policy..')
    allaccuracylist, allf1list, allpreclist, allreclist, allcountlist = [], [], [], [], []
    for tau in range(0,opt.episode):
        accuracylist, f1list, preclist, reclist, countlist = [], [], [], [], []
        train_iter, validation_iter, test_iter = dataset.batch_iter(opt.batch_size)
        model, optim = lib.train.create_model(opt, wordrepr)
        trainer = lib.train.Trainer(model, train_iter, validation_iter, optim, opt)
        _, _, _, _, acc, f1, prec, rec = trainer.train(opt.start_epoch, opt.end_epoch)[-1]
        accuracylist.append(acc)
        f1list.append(f1)
        preclist.append(prec)
        reclist.append(rec)
        countlist.append(dataset.count_new)
        logger.info('First accuracy: %.3f, f1: %.3f, prec: %.3f, rec: %.3f' % (acc, f1, prec, rec))

        new_examples = []
        #In every episode, run the trajectory
        for t in count():
            if(dataset.count_new >= opt.budget or len(dataset.train_unlabeled) < opt.nsamples): break
            #unlabeled_examples = dataset.sample_unlabeled(opt.k_num) #Random sample k points from D_pool
            #dataset.unlabeled_examples = unlabeled_examples
            dataset.unlabeled_examples = dataset.train_unlabeled
            logger.info('Episode:{}/{} Budget:{}/{} Unlabeled:{}'.format(str(tau+1),opt.episode, dataset.count_new, opt.budget, len(dataset.train_unlabeled)))
            query_strategy = lib.utils.choose_al_method(opt.al_method, dataset, model, opt)
            ask_xnew_active = query_strategy.make_query(n_samples=N_SAMPLES)

            if(len(ask_xnew_active)>0):
                for x_new in ask_xnew_active:
                    dataset.label(x_new)
                    #assert x_new.id[0] not in new_examples
                    new_examples.append(x_new.id[0])
                #print('new_examples', len(new_examples), new_examples)

                trainer = lib.train.Trainer(model, train_iter, validation_iter, optim, opt)
                _, _, _, _, acc, f1, prec, rec = trainer.train(opt.start_epoch, opt.end_epoch)[-1]

                #if((dataset.count_new+1) % opt.k_num*2 == 0):
                accuracylist.append(acc)
                f1list.append(f1)
                preclist.append(prec)
                reclist.append(rec)
                countlist.append(dataset.count_new)
                logger.info('accuracy: %.3f, f1: %.3f, prec: %.3f, rec: %.3f' % (acc, f1, prec, rec))

        dataset.reset()
        allaccuracylist.append(accuracylist)
        allf1list.append(f1list)
        allpreclist.append(preclist)
        allreclist.append(reclist)
        allcountlist.append(countlist)

    logger.info('Test finished.')
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    averageacc=list(np.mean(np.array(allaccuracylist), axis=0))
    averagef1=list(np.mean(np.array(allf1list), axis=0))
    averageprec=list(np.mean(np.array(allpreclist), axis=0))
    averagerec=list(np.mean(np.array(allreclist), axis=0))
    averagecount=list(np.mean(np.array(allcountlist), axis=0))

    #Save results to csv and plot!
    csv_results = os.path.join(opt.save_dir, '_'.join([opt.lang, opt.al_method, 'only_result.csv']))
    logging.info('Saving results to {}'.format(csv_results))
    lib.utils.save_results(csv_results, averageacc, averagef1, averageprec, averagerec, averagecount, 'only'+opt.al_method.upper())

if __name__ == '__main__':
    testing = opt.test.split(';')
    for test in testing:
        opt.test = test
        opt.episode =1
        main()


