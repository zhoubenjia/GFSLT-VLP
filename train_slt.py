# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right

# *user-defined
from models import gloss_free_model
from datasets import S2T_Dataset
import utils as utils

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER
try:
    from nlgeval import compute_metrics
except:
    print('Please install nlgeval package.')


# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler

# global definition
from definition import *

def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # *Drop out params
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * visualization
    parser.add_argument('--visualize', action='store_true')

    return parser

def main(args, config):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])

    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer = tokenizer, config=config, args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem)
    
    
    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer = tokenizer, config=config, args=args, phase='val')
    print(dev_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)
    
    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer = tokenizer, config=config, args=args, phase='test')
    print(test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)
    
    print(f"Creating model:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'], src_lang = 'de_DE', tgt_lang = 'de_DE')
    model = gloss_free_model(config, args)
    model.to(device)
    print(model)

    if args.finetune:
        print('***********************************')
        print('Load parameters for Visual Encoder...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if 'conv_2d' in k or 'conv_1d' in k:
                k = 'backbone.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v
            if 'trans_encoder' in k:
                k = 'mbart.model.encoder.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v
            
            if 'text_decoder' in state_dict:
                for k, v in state_dict['text_decoder'].items():
                    if 'decoder' in k:
                        k = 'mbart.model.decoder.'+'.'.join(k.split('.')[2:])
                        new_state_dict[k] = v
            
        # *replace the word embedding
        model_dict = torch.load(config['model']['transformer']+'/pytorch_model.bin', map_location='cpu')
        for k, v in model_dict.items():
            if 'decoder.embed_tokens.weight' in k:
                k = 'mbart.' + k
                new_state_dict[k] = v
            if 'decoder.embed_positions.weight' in k:
                k = 'mbart.' + k
                new_state_dict[k] = v

        ret = model.load_state_dict(new_state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)

    lr_scheduler = scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=1e-8,
                T_max=args.epochs,
            )
    loss_scaler = NativeScaler()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.2, num_classes=2454)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    output_dir = Path(args.output_dir)
    if args.resume:
        print('Resuming Model Parameters... ')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} ")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, loss_scaler, mixup_fn)
        lr_scheduler.step(epoch)

        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
        
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")

        if max_accuracy < test_stats["belu4"]:
            max_accuracy = test_stats["belu4"]
            if args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max BELU-4: {max_accuracy:.2f}%')
        if utils.is_main_process():
            wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'], 'dev/Bleu_4':test_stats['belu4'], 'dev/Best_Bleu_4': max_accuracy})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")
        
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, mixup_fn=None, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        out_logits = model(src_input, tgt_input)
        label = tgt_input['input_ids'].reshape(-1)
        logits = out_logits.reshape(-1,out_logits.shape[-1])
        loss = criterion(logits, label.to(device, non_blocking=True))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_mbart=round(float(optimizer.param_groups[1]["lr"]), 8))

        if (step+1) % 10 == 0 and args.visualize and utils.is_main_process():
            utils.visualization(model.module.visualize())
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion,  config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, 10, header)):

            out_logits = model(src_input, tgt_input)
            total_loss = 0.0
            label = tgt_input['input_ids'].reshape(-1)
            
            logits = out_logits.reshape(-1,out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))
            
            total_loss += tgt_loss

            metric_logger.update(loss=total_loss.item())
            
            output = model_without_ddp.generate(src_input, max_new_tokens=150, num_beams = 4,
                        decoder_start_token_id=tokenizer.lang_code_to_id['de_DE']
                        )

            tgt_input['input_ids'] = tgt_input['input_ids'].to(device)
            for i in range(len(output)):
                tgt_pres.append(output[i,:])
                tgt_refs.append(tgt_input['input_ids'][i,:])
            
            if (step+1) % 10 == 0 and args.visualize and utils.is_main_process():
                utils.visualization(model_without_ddp.visualize())

    pad_tensor = torch.ones(200-len(tgt_pres[0])).to(device)
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)
    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=PAD_IDX)

    pad_tensor = torch.ones(200-len(tgt_refs[0])).to(device)
    tgt_refs[0] = torch.cat((tgt_refs[0],pad_tensor.long()),dim = 0)
    tgt_refs = pad_sequence(tgt_refs,batch_first=True,padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)

    bleu = BLEU()
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score
    # metrics_dict['belu4']=bleu_s

    metric_logger.meters['belu4'].update(bleu_s)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* BELU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.belu4, losses=metric_logger.loss))
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+'/tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+'/tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')
        print('\n'+'*'*80)
        metrics_dict = compute_metrics(hypothesis=args.output_dir+'/tmp_pres.txt',
                           references=[args.output_dir+'/tmp_refs.txt'],no_skipthoughts=True,no_glove=True)
        print('*'*80)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
    if utils.is_main_process():
        wandb.init(project='GF-SLT',config=config)
        wandb.run.name = args.output_dir.split('/')[-1]
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("dev/*", step_metric="epoch")
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)

