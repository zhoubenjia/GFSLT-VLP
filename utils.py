"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time,random
import numpy as np
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid
import cv2

import torch.nn.functional as nnf
from einops import rearrange, repeat
import pickle
import gzip

try:
    from torchtext.vocab import build_vocab_from_iterator
except:
    pass
from itertools import groupby
import tensorflow as tf

import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
from torchvision.utils import save_image, make_grid

# global definition
from definition import *

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def count_parameters_in_MB(model):
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)

def cosine_scheduler(base_value, final_value, epochs):
    iters = np.arange(epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def cosine_scheduler_func(base_value, final_value, iters, epochs):
    schedule = lambda x: final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * x / epochs))
    return schedule(iters)

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def build_vocab(file_path,UNK_IDX,specials_symbols):
    vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=specials_symbols,min_freq=1)
    vocab.set_default_index(UNK_IDX)
    return vocab

def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield line.strip().split()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather,dim=0)
    return output

def gloss_tokens_to_sequences(tokens,tgt_vocab,type = 'tensor'):
    if type=='list':
        sequences = []
        for token in tokens:
            sequence = tgt_vocab.lookup_tokens(token)
            sequence = ' '.join(sequence)
            sequences.append(sequence)
        return sequences
    else:
        tokens = tokens.transpose(0,1)
        sequences = []
        for i in range(len(tokens)):
            token =  tokens[i,:].tolist()
            for j1 in range(len(token)):
                if token[j1] == PAD_IDX:
                    token = token[0:j1]
                    break
                if j1 == len(token)-1:
                    token = token[0:j1]
            sequence = tgt_vocab.lookup_tokens(token)
            sequence = ' '.join(sequence)
            sequences.append(sequence)
        return sequences

def NoiseInjecting(raw_gloss, noise_rate=0.15, noise_type='omit_last', random_shuffle=False, is_train=True):
    new_gloss = []

    for ii, gloss in enumerate(raw_gloss):
        text = gloss.split()

        if noise_type == 'omit':
            # del noise
            if random.uniform(0, 1) <= 1. and is_train:
                index = sampler_func(len(text), int(len(text)*(1. - noise_rate)), random_choice=is_train)
                noise_gloss = []
                noise_idx = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(i)
            else:
                noise_gloss = [d for d in text]

        elif noise_type == 'omit_last' :
            if random.uniform(0, 1) <= 1.0 and is_train:
                index = np.arange(0, len(text) - int(np.ceil(len(text)*(np.random.uniform(0,noise_rate,(1,))))), 1, dtype=int)
                noise_gloss = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
            else:
                noise_gloss = [d for d in text]
        
        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss) # random shuffle sequence

        new_gloss.append(' '.join(noise_gloss))
    return new_gloss

def GlossPadding(input_ids, gt_gloss, attention_mask):
    new_input_ids, new_gt_gloss, new_mask = [], [], []
    for NG, TG, MASK in zip(input_ids, gt_gloss, attention_mask):
        if len(NG) > len(TG):
            while len(NG) != len(TG):
                TG.append(1)
        if len(NG) < len(TG):
            while len(NG) != len(TG):
                NG.append(1)
                MASK.append(0)
        new_input_ids.append(NG)
        new_gt_gloss.append(TG)
        new_mask.append(MASK)
    return torch.tensor(new_input_ids), torch.tensor(new_gt_gloss), torch.tensor(new_mask)

def ctc_decode(gloss_probabilities,sgn_lengths):
    gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
    # tf_gloss_probabilities = np.concatenate(
    #     (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
    #     axis=-1,
    # )

    # ctc_decode, _ = tf.nn.ctc_greedy_decoder(
    #     inputs=gloss_probabilities,
    #     sequence_length=np.array(sgn_lengths),
    #     blank_index=SI_IDX,
    #     merge_repeated = False
    # )
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                        inputs=gloss_probabilities,
                        sequence_length=np.array(sgn_lengths),
                        beam_width=5,
                        top_paths=1,
                        )
    ctc_decode = ctc_decode[0]
    # Create a decoded gloss list for each sample
    tmp_gloss_sequences = [[] for i in range(gloss_probabilities.shape[1])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        if ctc_decode.values[value_idx].numpy() != SI_IDX:
            tmp_gloss_sequences[dense_idx[0]].append(
                ctc_decode.values[value_idx].numpy()
            )
    
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences

def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    if is_train:
        left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
    else:
        left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

    return (left, top, left + crop_size, top + crop_size), resize

class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 300
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]

def visualization(atten_maps):
    os.makedirs('./demo', exist_ok=True)
    for ii, att in enumerate(atten_maps):
        i = att.shape[0]
        idx = [max(1, int((i**0.5))), i//max(1, int((i**0.5))), 1]

        fig = plt.figure(figsize=(6*idx[1], 6*idx[0]))
        if att.squeeze().dim() == 2:
            ax = fig.add_subplot()
            att = torch.softmax(att, dim=-1)
            sns.heatmap(att.detach().cpu().numpy(), annot=False, yticklabels=False, xticklabels=False, fmt='g', ax=ax)
            
            fig.savefig(os.path.join('./demo', f'Att_score_{ii}.jpg'), dpi=fig.dpi)
            plt.close()
            continue
        
        for cmp in att:
            ax = fig.add_subplot(*idx)
            sns.heatmap(cmp.detach().cpu().numpy(), cbar=idx[-1] % idx[-2] == 0, annot=False, yticklabels=False, xticklabels=False, fmt='g', ax=ax)
            idx[-1] += 1
        fig.savefig(os.path.join('./demo', f'Att_score_{ii}.jpg'), dpi=fig.dpi)
        plt.close()

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=torch.nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
        
def loss_fn_kd(outputs, teacher_outputs, T=1.0, alpha=0.5):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = torch.nn.KLDivLoss( reduction='sum')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (T * T) #+ \
            #    F.cross_entropy(outputs, F.softmax(teacher_outputs, dim=1)) * (1. - alpha)

    return KD_loss

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def InputMask(gloss_input_ids, gloss_attention_mask, noise_rate=0.1, is_train=True):
    mask_matrix = torch.ones_like(gloss_attention_mask)
    for i in range(mask_matrix.size(0)):
        # index = random.sample(range(0, mask_matrix.size(-1)), int(mask_matrix.size(-1)*noise_rate))
        sample = sampler_func(mask_matrix.size(-1)-2, int((mask_matrix.size(-1)-2)*noise_rate), random_choice=is_train)
        index = [i+1 for i in sample]
        mask_matrix[i, :].scatter_(0, torch.tensor(index, device=mask_matrix.device), 0)
    gloss_attention_mask *= mask_matrix.cuda().type(torch.int)
    gloss_input_ids = torch.where(mask_matrix==0, torch.ones_like(gloss_input_ids), gloss_input_ids)
    # print(gloss_input_ids, gloss_attention_mask)

    # gloss_input_ids_filp = gloss_input_ids.flip(0)
    # mask_matrix = torch.ones_like(gloss_attention_mask)
    # for i in range(mask_matrix.size(0)):
    #     index = random.sample(range(0, mask_matrix.size(-1)), int(mask_matrix.size(-1)*noise_rate))
    #     mask_matrix[i, :].scatter_(0, torch.tensor(index, device=mask_matrix.device), 0)
    # gloss_input_ids = torch.where(mask_matrix==0, gloss_input_ids_filp, gloss_input_ids)


    return gloss_input_ids, gloss_attention_mask

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__  # dict.k  ==>  dict[k]
    # __getattr__ = dict.get  # dict.k  ==>  dict.get(k)
    # __getattr__ = lambda d, k: d.get(k, '')  # dict.k  ==>  dict.get(k,default)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False  

def save_dataset_file(path, data):
    with gzip.open(path, "w") as f:
        pickle.dump(data,f)

def param_groups_weight_decay(
        model: torch.nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]