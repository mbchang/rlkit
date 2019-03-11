import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import shutil
import torch

plt.style.use('ggplot')

def printf(logger, args, string):
    if args.printf:
        f = open(logger.logdir+'.txt', 'a')
        print(string, file=f)
    else:
        print(string)

def mkdirp(logdir):
    if '_debug' in logdir:
        # overwrite
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    else:
        try:
            os.mkdir(logdir)
        except FileExistsError:
            overwrite = 'o'
            while overwrite not in ['y', 'n']:
                overwrite = input('{} exists. Overwrite? [y/n] '.format(logdir))
            if overwrite == 'y':
                shutil.rmtree(logdir)
                os.mkdir(logdir)
            else:
                raise FileExistsError

class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = (1-self.alpha) * self.data['running_'+key] + self.alpha * value
        return copy.deepcopy(self.data['running_'+key])

    def get_value(self, key):
        if 'running_'+key in self.data:
            return self.data['running_'+key]
        else:
            assert KeyError

class Logger(object):
    def __init__(self, expname, logdir, params, variables=None, resumed_from=None):
        super(Logger, self).__init__()
        self.data = {}
        self.metrics = {}

        self.expname = expname
        self.logdir = logdir
        self.params = params
        self.resumed_from = resumed_from if resumed_from else None

        if self.resumed_from:
            assert os.path.exists(self.resumed_from)
            if os.path.dirname(self.resumed_from) != self.logdir:
                mkdirp(self.logdir)
        else:
            mkdirp(self.logdir)

        if variables is not None:
            self.add_variables(variables)

    def set_expname(self, expname):
        self.expname = expname

    def set_resumed_from(self, resume_path):
        self.data['resumed_from'] = resume_path

    #############################################
    def load_params_eval(self, eval_, resume):
        """ saved_args is mutable! """
        assert self.resumed_from is not None
        saved_args = self.load_params()
        saved_args.eval = eval_
        saved_args.resume = resume
        self.set_resumed_from(self.resumed_from)
        return saved_args

    def load_params_transfer(self, transfer, resume):
        """ saved_args is mutable! """
        assert self.resumed_from is not None
        saved_args = self.load_params()
        saved_args.transfer = transfer
        saved_args.resume = resume
        self.set_resumed_from(self.resumed_from)
        return saved_args

    # should be able to combine the above
    #############################################

    def save_params(self, logdir, params, ext=''):
        pickle.dump(params, open(os.path.join(self.logdir, '{}.p'.format('params'+ext)), 'wb'))

    def set_params(self, params):
        """ params is mutable """
        self.params = params

    def set_and_save_params(self, logdir, params, ext=''):
        self.set_params(params)
        self.save_params(logdir, params, ext)

    def load_params(self):
        params = pickle.load(open(os.path.join(self.logdir, '{}.p'.format('params')), 'rb'))
        return params

    def add_variables(self, names):
        for name in names:
            self.add_variable(name)

    def update_variables(self, name_values):
        for name, value in name_values:
            self.update_variable(name, value)

    def add_variable(self, name):
        self.data[name] = []

    def update_variable(self, name, value):
        self.data[name].append(value)

    def add_metric(self, name, initial_val, comparator):
        self.metrics[name] = {'value': initial_val, 'cmp': comparator}

    def save_checkpoint(self, ckpt_data, current_metrics, i_episode, args, ext):
        old_ckpts = [x for x in os.listdir(self.logdir) if '.pth.tar' in x and 'best' in x and ext in x]
        assert len(old_ckpts) <= len(current_metrics)

        for m in self.metrics:
            if self.metrics[m]['cmp'](current_metrics[m], self.metrics[m]['value']):
                self.metrics[m]['value'] = current_metrics[m]
                if any(m in oc for oc in old_ckpts):
                    old_ckpts_with_metric = [x for x in old_ckpts if 'best{}'.format(m) in x]
                    assert len(old_ckpts_with_metric) == 1
                    old_ckpt_to_remove = os.path.join(self.logdir,old_ckpts_with_metric[0])
                    os.remove(old_ckpt_to_remove)
                    printf(self, args, 'Removing {}'.format(old_ckpt_to_remove))
                torch.save(ckpt_data, os.path.join(self.logdir, '{}_ep{:.0e}_best{}{}.pth.tar'.format(
                    self.expname, i_episode, m, ext)))
                printf(self, args, 'Saved Checkpoint for best {}'.format(m))
            else:
                printf(self, args, 'Did not save {} checkpoint at because {} was worse than the best'.format(m, m))

    def plot(self, var1, var2, fname):
        plt.plot(self.data[var1], self.data[var2])
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.savefig(os.path.join(self.logdir,'{}.png'.format(fname)))
        plt.clf()

    def to_cpu(self, state_dict):
        cpu_dict = {}
        for k,v in state_dict.items():
            cpu_dict[k] = v.cpu()
        return cpu_dict

    def save(self, name):
        pickle.dump(self.data, open(os.path.join(self.logdir,'{}.p'.format(name)), 'wb'))

    def load(self, name):
        self.data = pickle.load(open(os.path.join(self.logdir,'{}.p'.format(name)), 'rb'))

def create_logger(build_expname, args):
    if args.resume:
        """
            - args.resume identifies the checkpoint that we will load the model
            - We will load the args from the saved checkpoint and overwrite the 
            default args.
            - The only things we will not overwrite is args.eval and args.resume,
            which have been provided by the current run
            - We will also set the resumed_from attribute of logger to point to
            the current checkpoint we just loaded up.
            - TODO
                it's an open question of whether we should resave the params
                again, which would now contain values for args.eval and 
                args.resume
        """
        if args.eval:
            logdir = os.path.dirname(args.resume)
            logger = Logger(
                expname='',  # will overwrite
                logdir=logdir,
                params={},  # will overwrite
                resumed_from=args.resume)
            args = logger.load_params_eval(args.eval, args.resume)
            expname = build_expname(args) + '_eval'
            logger.set_expname(expname)
            logger.save_params(logger.logdir, args, ext='_eval')
        elif args.transfer:
            expname = build_expname(args) + '_transfer'
            logger = Logger(
                expname=expname,
                logdir=os.path.join(args.outputdir, expname),
                params=args,
                resumed_from=args.resume)
            logger.save_params(logger.logdir, args, ext='_transfer')
        else:
            assert False, 'You tried to resume but you did not specify whether we are in eval or transfer mode'
    else:
        expname = build_expname(args)   
        logger = Logger(
            expname=expname, 
            logdir=os.path.join(args.outputdir, expname), 
            params=args, 
            resumed_from=None)
        logger.save_params(logger.logdir, args)
    return logger

