
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..", os.getcwd())))

import torch
from datasets import S2T_Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import utils as utils
import argparse
from collections import OrderedDict
from tqdm import tqdm
import yaml

from backup.c2rl_model import  C2RL_Pretrain
from backup.s2t_model import S2T_Model



def build_data(config):

    print(f"Creating dataset:")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])

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
    
    return train_dataloader, dev_dataloader, test_dataloader


def feature_ex(data_dataloder,model, label, config, type):
    if not os.path.exists(args.save_path + '/features'):
        os.mkdir(args.save_path + '/features')
    model.eval()

    fature_dict = OrderedDict()

    for src_input, tgt_input in tqdm(data_dataloder):
        with torch.no_grad():
            features = model.module.feature_ex(src_input)

        for idx, name in enumerate(src_input['name_batch']):
            feature = features[idx]
            feature = feature.cpu()
            # print(name, feature.shape)
            fature_dict[name] = feature

    return fature_dict

def main(config):

    train_dataloader, dev_dataloader, test_dataloader = build_data(config)

    print(f"Creating model:")
    # model = C2RL_Pretrain(
    #     config=config
    #     )
    model = S2T_Model(config)
    print(model)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cuda(args.local_rank))
    
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=False)
    

    state_dict = torch.load(args.save_path + '/best_model.pth', map_location='cpu')
    
    ret = model.module.load_state_dict(state_dict['model'], strict=True)

    print('successfully load ' + args.save_path + '/best_checkpoint.pth')


    dev_label = utils.load_dataset_file(config['data']['dev_label_path'])
    train_label = utils.load_dataset_file(config['data']['train_label_path'])
    test_label = utils.load_dataset_file(config['data']['test_label_path'])
    
    train = feature_ex(train_dataloader,model, train_label, config, 'train')
    torch.distributed.barrier()
    # print(train)
    dev = feature_ex(dev_dataloader,model, dev_label, config, 'dev')
    torch.distributed.barrier()
    # print(dev)
    
    test = feature_ex(test_dataloader,model, test_label, config, 'test')
    torch.distributed.barrier()
    # print(test)
    
    if args.local_rank == 0:
        dev_label = utils.load_dataset_file(config['data']['dev_label_path'])
        train_label = utils.load_dataset_file(config['data']['train_label_path'])
        test_label = utils.load_dataset_file(config['data']['test_label_path'])
        for k ,v in dev_label.items():
            dev_label[k]['feature'] = dev[k]
        for k ,v in test_label.items():
            test_label[k]['feature'] = test[k]
        for k ,v in train_label.items():
            train_label[k]['feature'] = train[k]
        utils.save_dataset_file(args.save_path + '/features/features.dev', dev_label)
        utils.save_dataset_file(args.save_path + '/features/features.test', test_label)
        utils.save_dataset_file(args.save_path + '/features/features.train', train_label)




def make_new(label,path):
    new_dev_label = {}
    for key,sample in label.items():
        sample['feature'] = torch.ones(1,1)
        sample['prediction'] = torch.ones(1,1)
    return label
    

if __name__ == '__main__':

    utils.set_seed(42)
    parse = argparse.ArgumentParser("slt")
    parse.add_argument('--local_rank', default=0, type=int)
    parse.add_argument('--config_path', default='/homedata/bjzhou/codes/VLP/GFSLT-VLP/configs/config_gloss_free.yaml',type=str)
    parse.add_argument('--save-path', required=True, default='', type=str)
    parse.add_argument('--batch-size', default=4, type=int)
    parse.add_argument('--num_workers', default=8, type=int)
    parse.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parse.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parse.set_defaults(pin_mem=True)

        # * data process params
    parse.add_argument('--input-size', default=224, type=int)
    parse.add_argument('--resize', default=256, type=int)


    args = parse.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    with open(args.config_path, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    main(config)