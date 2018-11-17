#__date__ = 6/14/18
#__time__ = 4:09 PM
#__author__ = isminilourentzou

import lib
import torch
import os
import logging

logger = logging.getLogger("model")


def build_wordrepr(opt, vocabs):
    if opt.load_from is not None:
        checkpoint = get_checkpoint(opt.load_from, opt)
        model = lib.model.WordRepr(checkpoint['opt'], vocabs)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = lib.model.WordRepr(opt, vocabs)
    if opt.cuda: model.cuda()  # GPU.
    return model


def build_active_learner(opt, wordrepr, network='dqn'):
    if(network=='cluster'):
        policy = lib.model.Cluster(opt, wordrepr)
    elif(network=='dqn'):
        policy = lib.model.DQN(opt, wordrepr)
    elif(network=='imitation'):
        policy = lib.model.Policy(opt, wordrepr)
    elif(network=='a2c'):
        policy = lib.model.ActorCritic(opt, wordrepr)
    else:
        raise NotImplementedError('Choices for AL network are: dqn, policy, a2c')
    optim = _create_optim(policy, opt)
    if opt.cuda: policy.cuda()  # GPU.
    return policy, optim

def create_active_learner(opt, wordrepr, network='dqn', load_from=None):
    if load_from is not None:
        checkpoint = get_checkpoint(os.path.join(opt.save_dir, load_from), opt)
        model, optim = build_active_learner(checkpoint['opt'], wordrepr, network)
        load_checkpoint(checkpoint, model, optim, opt)
    else:
        model, optim = build_active_learner(opt, wordrepr, network)
    finish_creation(opt, model)
    return model, optim

def build_model(opt, wordrepr):
    model = lib.model.Model(opt, wordrepr)
    optim = _create_optim(model, opt)
    if opt.cuda: model.cuda()  # GPU.
    return model, optim

def create_model(opt, wordrepr):
    if(opt.plainCRF):
        if opt.load_from:
            model = lib.model.CRFTagger(opt, wordrepr, model_file=opt.load_from)
            model.tagger.open(opt.load_from)
        else:
            model_file = os.path.join(opt.save_dir, opt.lang +'_'+'CRFplain.pt')
            model = lib.model.CRFTagger(opt, wordrepr, model_file)
        optim = None
    else:
        if opt.load_from is not None:
            checkpoint = get_checkpoint(opt.load_from, opt)
            model, optim = build_model(checkpoint['opt'], wordrepr)
            load_checkpoint(checkpoint, model, optim, opt)
        else:
            model, optim = build_model(opt, wordrepr)
        finish_creation(opt, model)
    return model, optim

def _create_optim(model, opt):
    trained_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = lib.train.Optim(
        trained_params, opt.optim, opt.lr, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_after=opt.start_decay_after
    )
    return optim

def get_checkpoint(load_from, opt):
    logger.info('Loading model from checkpoint at {}'.format(load_from))
    if opt.cuda:
        location = lambda storage, loc: storage.cuda(opt.gpu)
    else:
        location = lambda storage, loc: storage
    checkpoint = torch.load(load_from, map_location=location)
    checkpoint['opt'].cuda = opt.cuda
    return checkpoint

def load_checkpoint(checkpoint, model, optim, opt):
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])

def finish_creation(opt, model):
    if opt.cuda: model.cuda() # GPU.
    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('* number of parameters: %d' % nParams)
    logger.info(model)