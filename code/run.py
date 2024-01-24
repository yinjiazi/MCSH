import os
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from config.config_run import Config
from config.config_debug import ConfigDebug
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader, ten_fold_split_v2, get_splited_ten_fold_files

# logger = logging.getLogger('MMSA')
#
#
# def _set_logger(log_dir, model_name, dataset_name):
#     log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
#     logger = logging.getLogger('MMSA')
#     logger.setLevel(logging.INFO)
#     handler = logging.FileHandler(log_file_path)
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#
#     return logger


# log_dir = Path.home() / "mmsa" / "logs"
# logger = _set_logger(log_dir, model_name='miat', dataset_name='mcsh')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def str2listoffints(v):
    temp_list = v.split('=')
    temp_list = [list(map(int, t.split("-"))) for t in temp_list]
    return temp_list


def str2bool(v):
    """string to boolean"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError("Boolean value expected." + v)


def str2bools(v):
    return list(map(str2bool, v.split("-")))


def str2floats(v):
    return list(map(float, v.split("-")))


def run(args):
    # if args.log_dir:
    #     log_dir = args.log_dir
    # else:  # use default log save dir
    #     log_dir = Path.home() / "mmsa" / "logs"
    # Path(log_dir).mkdir(parents=True, exist_ok=True)
    # logger = _set_logger(log_dir, model_name='miat', dataset_name='mcsh')

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # device
    # using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    # using_cuda = False
    device = torch.device('cuda:%s' % args.gpu_ids)
    print(f"Let's use {device}")
    args.device = device
    # data
    ten_flod_results = []
    for dataloader in get_splited_ten_fold_files(args):
        model = AMIO(args).to(device)
        # using multiple gpus
        # if using_cuda and len(args.gpu_ids) > 1:
        # model = torch.nn.DataParallel(model)
        #                                   device_ids=args.gpu_ids,
        #                                   output_device=args.gpu_ids[3])
        # start running
        # do train
        atio = ATIO().getTrain(args)
        # do train
        atio.do_train(model, dataloader)
        # load pretrained model
        pretrained_path = os.path.join(args.model_save_path, \
                                       f'{args.modelName}-{args.datasetName}-{args.tasks}.pth')
        assert os.path.exists(pretrained_path)
        model.load_state_dict(torch.load(pretrained_path))
        model.to(device)
        # do test
        if args.modelName == 'miat':
            results, val_results = atio.do_test(model, dataloader['test'], mode="TEST")
            ten_flod_results.append(results)
        else:
            results = atio.do_test(model, dataloader['test'], mode="TEST")
            ten_flod_results.append(results)

    class multidict(dict):
        def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    def avg_ten_fold_results(ten_flod_results):

        if len(ten_flod_results[0][args.tasks].keys()) == 2:
            res = multidict()
            for type in ['sarcasm', 'humor']:
                criterions = list(ten_flod_results[0][args.tasks][type].keys())
                for c in criterions:
                    values = [r[args.tasks][type][c] for r in ten_flod_results]
                    mean = np.mean(values)
                    res[args.tasks][type][c] = mean
        else:
            res = multidict()
            criterions = list(ten_flod_results[0][args.tasks].keys())
            for c in criterions:
                values = [r[args.tasks][c] for r in ten_flod_results]
                mean = np.mean(values)
                res[args.tasks][c] = mean

        return res

    ten_flod_avg_results = avg_ten_fold_results(ten_flod_results)
    # else:
    #     results = atio.do_test(model, dataloader['test'], mode="TEST")
    return ten_flod_avg_results


def run_debug(seeds, debug_times=50):
    print('You are using DEBUG mode!')
    for i in range(debug_times):
        # cancel random seed
        args = parse_args()
        setup_seed(int(time.time()))
        config = ConfigDebug(args)
        args = config.get_config()
        # print debugging params
        print("#" * 40 + '%s-(%d/%d)' % (args.modelName, i + 1, debug_times) + '#' * 40)
        for k, v in args.items():
            if k in args.d_paras:
                print(k, ':', v)
        print("#" * 90)
        print('Start running %s...' % (args.modelName))
        results = []
        for j, seed in enumerate(seeds):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args)[args.tasks[0]])
        # save results to csv
        print('Start saving results...')
        if not os.path.exists(args.res_save_path):
            os.makedirs(args.res_save_path)
        # load resutls file
        save_file_path = os.path.join(args.res_save_path, \
                                      args.datasetName + '-' + args.modelName + '-' + args.tasks + '-debug.csv')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns=[k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))
        # save results
        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        print('Results are saved to %s...' % (save_file_path))


def run_normal(tasks):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # debug
    model_results = []
    # run results
    # for i, seed in enumerate(seeds):
    args = parse_args()
    # args.cur_time = i+1
    # load config
    config = Config(args)
    args = config.get_config()
    print(args)
    # print(args.hidden_dims)
    args.tasks = tasks
    # setup_seed(seed)
    # args['seed'] = seed
    print(f'Start running {args.modelName} {args.tasks}...')
    # runnning
    test_results = run(args)
    # restore results
    model_results.append(test_results)
    # save results
    if len(model_results[0][args.tasks].keys()) == 2:
        # for type in ['sarcasm','humor']:
        criterions = list(model_results[0][args.tasks][args.type].keys())
        # df = pd.DataFrame(columns=["Model"] + criterions)
        # res = [args.modelName+'-'+args.tasks]
        #
        #     for c in criterions:
        #         values = [r[args.tasks][type][c] for r in model_results]
        #         mean = round(np.mean(values)*100, 2)
        #         std = round(np.std(values)*100, 2)
        #         res.append((mean, std))
        #     df.loc[len(df)] = res
        df = pd.DataFrame(model_results)
        save_path = os.path.join(args.res_save_path, \
                                 args.modelName + '-' + args.tasks + '-' + \
                                 'saracasm' + '-' + str(
                                     round(model_results[0][args.tasks]['sarcasm']['f1-score'] * 100, 2)) + '-' + \
                                 'humor' + '-' + str(
                                     round(model_results[0][args.tasks]['humor']['f1-score'] * 100, 2)) + '.csv')
        if not os.path.exists(args.res_save_path):
            os.makedirs(args.res_save_path)
        df.to_csv(save_path, index=None)
        print('Results are saved to %s...' % (save_path))
    else:
        criterions = list(model_results[0][args.tasks].keys())
        df = pd.DataFrame(columns=["Model"] + criterions)
        res = [args.modelName + '-' + args.tasks]
        for c in criterions:
            values = [r[args.tasks][c] for r in model_results]
            mean = round(np.mean(values) * 100, 2)
            std = round(np.std(values) * 100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        save_path = os.path.join(args.res_save_path, \
                                 args.modelName + '-' + args.type + '-' + args.tasks + '-' + \
                                 str(round(model_results[0][args.tasks]['f1-score'] * 100, 2)) + '.csv')
        if not os.path.exists(args.res_save_path):
            os.makedirs(args.res_save_path)
        df.to_csv(save_path, index=None)
        print('Results are saved to %s...' % (save_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', type=bool, default=False,
                        help='adjust parameters ?')
    parser.add_argument('--modelName', type=str, default='attmi',
                        help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn/mtfn/mlmf/mlf_dnn')
    parser.add_argument('--datasetName', type=str, default='mcsh',
                        help='support mosi/sims/mcsh')
    parser.add_argument('--tasks', type=str, default='M',
                        help='M/T/A/V/MTAV/...')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                        help='path to save results.')
    # parser.add_argument('--data_dir', type=str, default='/home/sharing/disk3/dataset/multimodal-sentiment-dataset',
    #                     help='path to data directory')
    parser.add_argument('--data_dir', type=str, default='/home/xjnu1/mmsa',
                        help='path to data directory')
    parser.add_argument('--gpu_ids', type=int, default=0,
                        help='indicates the gpus will be used.')
    # parser.add_argument('--type', type=str, default='humor',
    #                     help='sarcasm/humor')
    parser.add_argument('--type', type=str, default='sarcasm',
                        help='sarcasm/humor')
    parser.add_argument('--alignNet', type=str, default='conv1d',
                        help='avg pool/ ctc / conv1d')

    # # cubemlp
    # parser.add_argument("--d_hiddens", default='10-2-64=5-2-32', type=str2listoffints)
    # parser.add_argument("--d_outs", default='10-2-64=5-2-32', type=str2listoffints)
    # parser.add_argument("--dropout_mlp", default='0.5-0.5-0.5', type=str2floats)
    # parser.add_argument("--dropout", default='0.5-0.5-0.5-0.5', type=str2floats)
    # parser.add_argument("--bias", action='store_true')
    # parser.add_argument("--ln_first", action='store_true')
    # parser.add_argument("--res_project", default='1-1', type=str2bools)

    return parser.parse_args()


if __name__ == '__main__':
    seeds = [1, 12, 123, 1234, 12345]
    import warnings

    warnings.filterwarnings("ignore")
    # seeds = [1]

    if parse_args().debug_mode:
        run_debug(seeds, debug_times=200)
    else:
        # for task in [ 'T', 'A', 'V','TA','TV', 'AV',  'M']:
        # for i in range(5):
        for task in ['M']:
            run_normal(task)
