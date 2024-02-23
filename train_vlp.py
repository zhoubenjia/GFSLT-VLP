
# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig

# *user-defined
from models import  SLRCLIP
import utils as utils
from datasets import S2T_Dataset

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
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger


# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image


from hpman.m import _
import hpargparse

# global definition
from definition import *



def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)


    # distributed training parameters
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
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
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
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * wandb params
    parser.add_argument("--log_all", action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )
    parser.add_argument("--entity", type=str, 
        help="wandb entity",
    )
    parser.add_argument("--project", type=str, default='VLP',
        help="wandb project",
    )

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
    cudnn.benchmark = False # Since the input dim is dynamic.

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
                                 pin_memory=args.pin_mem,
                                 drop_last=True)
    
    
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
    model = SLRCLIP(
        config=config
        )
    model.to(device)
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret =  model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    # create optimizer and scheduler
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = utils.KLLoss()
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, criterion, args.start_epoch)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, args.start_epoch)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, loss_scaler)
        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(args, dev_dataloader, model, criterion, epoch)

        if min_loss > test_stats["loss"]:
            min_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / f'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'args': args,
                    }, checkpoint_path)
        
        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")
        if utils.is_main_process():
            wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'], 'dev/min_loss': min_loss})


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
        torch.distributed.barrier()
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        dev_stats = evaluate(args, dev_dataloader, model, criterion, epoch)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, epoch)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text, ground_truth = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            total_loss = (loss_imgs + loss_texts)/2.
        loss_scaler(total_loss, optimizer)

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if (step+1) % 10 == 0 and utils.is_main_process():
            visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
            utils.visualization([visual_map,])

    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/train_loss':loss_value})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, criterion, epoch):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion

    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):

            logits_per_image, logits_per_text, ground_truth = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts)/2

            metric_logger.update(loss=total_loss.item())
            if (step+1) % 10 == 0 and utils.is_main_process():
                visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
                utils.visualization([visual_map, ])

    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/dev_loss':total_loss.item()})
    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False

    return run
if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)