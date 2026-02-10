from os import path

class my_logger:
    def __init__(self, args):
        name = "{}_{}_{}_{}_{}.log".format(args.name, args.network, args.dataset, args.train_attack_iters,
                                           args.attack_learning_rate)
        args.name = name
        self.name = path.join(args.log_dir, name)
        with open(self.name, 'w') as F:
            print('\n'.join(['%s:%s' % item for item in args.__dict__.items() if item[0][0] != '_']), file=F)
            print('\n', file=F)

    def info(self, content):
        with open(self.name, 'a') as F:
            print(content)
            print(content, file=F)
