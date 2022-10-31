
import os
import math
import random
import torch
import torch.nn.functional as f
from tqdm import tqdm
from .network import Network
from .utils.metrics import registry as metrics
import numpy as np
import torch.nn as nn
from pytorch_transformers import AdamW, WarmupLinearSchedule, BertTokenizer,WarmupCosineSchedule
from GMutil.factorize_graph_matching import kronecker_sparse, kronecker_torch
from GMutil.sparse_torch import CSRMatrix3d

from torch.autograd import Variable


class Model:
    prefix = 'checkpoint'
    best_model_name = 'best.pt'

    def __init__ (self, args, state_dict=None):
        self.args = args

        # network

        # print(self.network.blocks[0]['alignment'].alignparsing.ParserNet.parameters())
        # exit(0)
        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        # print(torch.cuda.current_device())
        self.network = Network(args,self.device)
        self.network.to(self.device)

        # optimizer
        self.params = list(filter(lambda x: x.requires_grad, self.network.parameters()))
        self.bert_params = list(map(id, self.network.bert_feature.parameters()))
        self.gcn_params = filter(lambda p: id(p) not in self.bert_params, self.network.parameters())

        self.params = [
            {"params": self.network.bert_feature.parameters(), "lr": args.lr},
            {"params": self.gcn_params, "lr": args.syn_lr},
        ]



        self.tokenizer = BertTokenizer.from_pretrained(args.bert_vocal_dir)
        self.opt = AdamW (self.params, args.lr, correct_bias=False)
        num_total_steps = args.epochs * int(np.ceil(args.total_data / args.batch_size))
        num_warmup_steps = num_total_steps * args.warmup_rate
        self.scheduler = WarmupLinearSchedule(self.opt,warmup_steps=num_warmup_steps, t_total=num_total_steps)

        self.losses = nn.CrossEntropyLoss()
        # updates
        self.Syntactic_losses = nn.CrossEntropyLoss()


        self.updates = state_dict['updates'] if state_dict else 0


    def _update_schedule (self):
        if self.args.lr_decay_rate < 1.:
            args = self.args
            t = self.updates
            base_ratio = args.min_lr / args.lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[0]['lr'] = args.lr * ratio

            base_ratio = args.gcn_min_lr / args.gcn_lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[1]['lr'] = args.gcn_lr * ratio

    def update (self, batches):
        self.network.train()

        self.opt.zero_grad()
        inputs, target = self.process_data(batches)
        output,syn_loss = self.network(inputs)
        summary = self.network.get_summary()
        #print(self.get_loss(output,target))
        loss = self.get_loss(output, target) + syn_loss
        loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(self.params[0]["params"],
                                                   self.args.grad_clipping) ) + (torch.nn.utils.clip_grad_norm_(self.params[1]["params"],self.args.grad_clipping) )


        self.opt.step()
        self.scheduler.step()
        self.updates += 1
        stats = {
            'updates': self.updates,
            'loss': loss.item(),
            'syn_loss':syn_loss.item(),
            #'loss2':loss2.item(),
            'lr': self.opt.param_groups[0]['lr'],
            'syn_lr': self.opt.param_groups[1]['lr'],
            'gnorm': grad_norm,
            'summary': summary,
        }
        return stats

    def evaluate (self, data):
        self.network.eval()
        targets = []
        probabilities = []
        predictions = []
        losses = []

        sparsity = []
        bert_time = []
        GM_time = []
        for dev_id in tqdm(range(int(np.floor(len(data) / self.args.batch_size)))):
            min_index = dev_id * self.args.batch_size
            max_index = min(len(data), (dev_id + 1) * self.args.batch_size)
            inputs, target = self.process_data(data[min_index:max_index])
            with torch.no_grad():
                output,syn_loss = self.network(inputs)
                loss = self.get_loss(output, target) + syn_loss
                pred = torch.argmax(output, dim=1)
                prob = torch.nn.functional.softmax(output, dim=1)
                losses.append(loss.item())
                targets.extend(target.tolist())
                probabilities.extend(prob.tolist())
                predictions.extend(pred.tolist())



        outputs = {
            'target': targets,
            'prob': probabilities,
            'pred': predictions,
            'args': self.args,
        }

        stats = {
            'updates': self.updates,
            #'loss': loss,
        }
        for metric in self.args.watch_metrics:
            if metric not in stats:  # multiple metrics could be computed by the same function
                stats.update(metrics[metric](outputs))
        assert 'score' not in stats, 'metric name collides with "score"'
        eval_score = stats[self.args.metric]
        stats['score'] = eval_score

        return eval_score, stats  # first value is for early stopping

    def predict (self, batch):
        self.network.eval()
        inputs, _ = self.process_data(batch)
        with torch.no_grad():
            output = self.network(inputs)
            output = torch.nn.functional.softmax(output, dim=1)
        return output.tolist()

    def process_data (self, batch):

        batch_size = len(batch)

        textX_batch = []
        textY_batch = []

        RelPOSXid_batch = []
        RelNERXid_batch = []
        RelPOSYid_batch = []
        RelNERYid_batch = []

        RelSynXid_batch = []
        RelSemXid_batch = []
        RelSynYid_batch = []
        RelSemYid_batch = []

        OneX_mask_batch = []
        OneY_mask_batch = []

        bert_full_batch = []
        bert_segment_batch = []
        bert_mask_batch = []
        SynXG1_batch = []
        SynXG2_batch = []
        SemXG1_batch = []
        SemXG2_batch = []
        SynYG1_batch = []
        SynYG2_batch = []
        SemYG1_batch = []
        SemYG2_batch = []

        X_length = []
        Y_length = []


        target = []
        for ids in range(batch_size):
            target.append(batch[ids]['target'])


            textX_batch.append(batch[ids]['textX'])
            textY_batch.append(batch[ids]['textY'])
            bert_full_batch.append([self.tokenizer.convert_tokens_to_ids("[CLS]")] + textX_batch[ids] + [self.tokenizer.convert_tokens_to_ids("[SEP]")]
                                       +textY_batch[ids] + [self.tokenizer.convert_tokens_to_ids("[SEP]")])
            bert_segment_batch.append([1]*(2+self.args.max_one_len)+[0]*(1+self.args.max_one_len))
            bert_mask_batch.append([1] + batch[ids]["OneX_mask_batch"] + [1] + batch[ids]["OneY_mask_batch"] + [1])

            RelPOSXid_batch.append(batch[ids]['RelOnePOSXid'])
            RelNERXid_batch.append(batch[ids]['RelOneNERXid'])
            RelPOSYid_batch.append(batch[ids]['RelOnePOSYid'])
            RelNERYid_batch.append(batch[ids]['RelOneNERYid'])


            RelSynXid_batch.append(batch[ids]['RelSynXid'])
            RelSemXid_batch.append(batch[ids]['RelSemXid'])
            RelSynYid_batch.append(batch[ids]['RelSynYid'])
            RelSemYid_batch.append(batch[ids]['RelSemYid'])

            OneX_mask_batch.append(batch[ids]['OneX_mask_batch'])
            OneY_mask_batch.append(batch[ids]['OneY_mask_batch'])

            SynXG1_batch.append(batch[ids]['SynXG1'])
            SynXG2_batch.append(batch[ids]['SynXG2'])
            SemXG1_batch.append(batch[ids]['SemXG1'])
            SemXG2_batch.append(batch[ids]['SemXG2'])
            SynYG1_batch.append(batch[ids]['SynYG1'])
            SynYG2_batch.append(batch[ids]['SynYG2'])
            SemYG1_batch.append(batch[ids]['SemYG1'])
            SemYG2_batch.append(batch[ids]['SemYG2'])



            X_length.append(batch[ids]['X_len'])
            Y_length.append(batch[ids]['Y_len'])


        #########sparse kronecker_sparse implement in NGM~\cite{NGM}
        Syn1G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(SynYG1_batch, SynXG1_batch)]
        Syn2G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(SynYG2_batch, SynXG2_batch)]
        Syn1G = CSRMatrix3d(Syn1G,device=self.device)
        Syn2G = CSRMatrix3d(Syn2G,device=self.device).transpose()

        Sem1G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(SemYG1_batch, SemXG1_batch)]
        Sem2G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(SemYG2_batch, SemXG2_batch)]
        Sem1G = CSRMatrix3d(Sem1G,device=self.device)
        Sem2G = CSRMatrix3d(Sem2G,device=self.device).transpose()

        #ret['KGHs'] = K1G, K1H

        RelPOSXid_batch = torch.LongTensor(RelPOSXid_batch).to(self.device)
        #print(RelPOSXid_batch.shape)
        RelNERXid_batch = torch.LongTensor(RelNERXid_batch).to(self.device)
        RelPOSYid_batch = torch.LongTensor(RelPOSYid_batch).to(self.device)
        RelNERYid_batch = torch.LongTensor(RelNERYid_batch).to(self.device)

        RelSynXid_batch = torch.LongTensor(RelSynXid_batch).to(self.device)
        RelSemXid_batch = torch.LongTensor(RelSemXid_batch).to(self.device)
        RelSynYid_batch = torch.LongTensor(RelSynYid_batch).to(self.device)
        RelSemYid_batch = torch.LongTensor(RelSemYid_batch).to(self.device)

        OneX_mask_batch = torch.FloatTensor(OneX_mask_batch).to(self.device)
        OneY_mask_batch = torch.FloatTensor(OneY_mask_batch).to(self.device)

        # SynXG1_batch = torch.FloatTensor(SynXG1_batch).to(self.device)
        # SynXG2_batch = torch.FloatTensor(SynXG2_batch).to(self.device)
        # SemXG1_batch = torch.FloatTensor(SemXG1_batch).to(self.device)
        # SemXG2_batch = torch.FloatTensor(SemXG2_batch).to(self.device)
        # SynYG1_batch = torch.FloatTensor(SynYG1_batch).to(self.device)
        # SynYG2_batch = torch.FloatTensor(SynYG2_batch).to(self.device)
        # SemYG1_batch = torch.FloatTensor(SemYG1_batch).to(self.device)
        # SemYG2_batch = torch.FloatTensor(SemYG2_batch).to(self.device)



        inputs = {
            'bert_full_batch':torch.LongTensor(bert_full_batch).to(self.device),
            'bert_segment_batch':torch.LongTensor(bert_segment_batch).to(self.device),
            'bert_mask_batch':torch.FloatTensor(bert_mask_batch).to(self.device),

            'SynG':(Syn1G,Syn2G),
            'SemG':(Sem1G,Sem2G),
            #

            'RelPOSXid_batch': RelPOSXid_batch,
            'RelNERXid_batch': RelNERXid_batch,
            'RelPOSYid_batch': RelPOSYid_batch,
            'RelNERYid_batch': RelNERYid_batch,
            'RelSynXid_batch': RelSynXid_batch,
            'RelSemXid_batch': RelSemXid_batch,
            'RelSynYid_batch': RelSynYid_batch,
            'RelSemYid_batch': RelSemYid_batch,
            'OneX_mask_batch': OneX_mask_batch,
            'OneY_mask_batch': OneY_mask_batch,

            # 'SynXG1_batch': SynXG1_batch,
            # 'SynXG2_batch': SynXG2_batch,
            # 'SemXG1_batch': SemXG1_batch,
            # 'SemXG2_batch': SemXG2_batch,
            # 'SynYG1_batch': SynYG1_batch,
            # 'SynYG2_batch': SynYG2_batch,
            # 'SemYG1_batch': SemYG1_batch,
            # 'SemYG2_batch': SemYG2_batch,

            'X_length':X_length,
            'Y_length':Y_length,
            'batch_size':batch_size,

        }

        # if 'target' in batch:
        # constract_target = np.zeros((batch_size,batch_size))
        # for i in range(batch_size):
        #     constract_target[i,i] = target[i]

        target = torch.LongTensor(target).to(self.device)
        #target = torch.FloatTensor(constract_target).to(self.device)

        # return inputs, target
        return inputs, target

    @staticmethod
    def get_loss (logits, target):
        #return f.binary_cross_entropy_with_logits(logits,target)
        return f.cross_entropy(logits, target)

    def save (self, states, name=None):
        if name:
            filename = os.path.join(self.args.summary_dir, name)
        else:
            filename = os.path.join(self.args.summary_dir, f'{self.prefix}_{self.updates}.pt')
        params = {
            'state_dict': {
                'model': self.network.state_dict(),
                'opt': self.opt.state_dict(),
                'updates': self.updates,
            },
            'args': self.args,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state()
        }
        params.update(states)
        if self.args.cuda:
            params['torch_cuda_state'] = torch.cuda.get_rng_state()
        torch.save(params, filename)

    @classmethod
    def load (cls, file):
        checkpoint = torch.load(file, map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ))
        prev_args = checkpoint['args']

        # update args
        prev_args.output_dir = os.path.dirname(os.path.dirname(file))
        prev_args.summary_dir = os.path.join(prev_args.output_dir, prev_args.name)
        prev_args.cuda = prev_args.cuda and torch.cuda.is_available()
        return cls(prev_args, state_dict=checkpoint['state_dict']), checkpoint

    def num_parameters (self):
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        return num_params