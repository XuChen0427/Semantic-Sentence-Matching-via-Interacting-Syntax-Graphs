
import os
import sys
import json5
from pprint import pprint
from src.utils import params
from src.trainer import Trainer
import torch
import numpy as np
import random
seed = 2021

torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def main():
    argv = sys.argv
    torch.autograd.set_detect_anomaly(True)
    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])
        for args, config in arg_groups:
            trainer = Trainer(args)
            states = trainer.train()
            with open('models/log.jsonl', 'a') as f:
                f.write(json5.dumps({
                    'data': os.path.basename(args.data_dir),
                    'params': config,
                    'state': states,
                }))
                f.write('\n')
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print('Usage: "python train.py configs/xxx.json5"')


if __name__ == '__main__':
    main()
