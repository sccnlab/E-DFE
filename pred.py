import os
import errno
import logging
from contextlib import contextmanager
from typing import Any, Iterable, Union, List, Tuple, Dict, Callable, Optional
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import models, utils
from torch import Tensor, nn, optim
from torch.utils import data
from tqdm import tqdm, tqdm_notebook
from nest import register, Context
import scipy.signal
import numpy as np


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

@register
def network_pred(
    data_loaders: Tuple[List[Tuple[str, data.DataLoader]], List[Tuple[str, data.DataLoader]]],
    model: nn.Module,
    criterion: object,
    optimizer: Callable[[Iterable], optim.Optimizer],
    parameter: Optional[Callable] = None,
    meters: Optional[Callable] = None, #Optional[Dict[str, Callable[[Context], Any]]] = None,
    hooks: Optional[Dict[str, List[Callable[[Context], None]]]] = None,
    max_epoch: int = 200,
    test_interval: int = 1,
    resume: Optional[str] = None,
    saved_path: Optional[str] = None,
    log_path: Optional[str] = None,
    device: str = 'cuda',
    use_data_parallel: bool = True,
    use_cudnn_benchmark: bool = True,
    random_seed: int = 999) -> Context:
    """Network trainer.
    """

    torch.manual_seed(random_seed)

    logger = logging.getLogger('nest.network_trainer')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    screen_handler = TqdmHandler()
    screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(screen_handler)

    if not log_path is None:
        try:
            os.makedirs(os.path.dirname(log_path))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        file_handler = logging.FileHandler(log_path, encoding='utf8')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

    def run_in_notebook():
        try:
            return get_ipython().__class__.__name__.startswith('ZMQ')
        except NameError:
            pass
        return False
    progress_bar = tqdm_notebook if run_in_notebook() else tqdm

    device = torch.device(device)
    if device.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is not available.'
        torch.backends.cudnn.benchmark = use_cudnn_benchmark

    train_loaders, test_loaders = data_loaders

    model = model.to(device)
    if device.type == 'cuda' and use_data_parallel:
        model = nn.DataParallel(model)

    params = model.parameters() if parameter is None else parameter(model)
    optimizer = optimizer(params)

    start_epoch_idx = 0
    start_batch_idx = 0
    if not resume is None:
        logger.info('loading checkpoint "%s"' % resume)
        checkpoint = torch.load(resume)
        start_epoch_idx = checkpoint['epoch_idx']
        start_batch_idx = checkpoint['batch_idx']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('checkpoint loaded (epoch %d)' % start_epoch_idx)



    ctx = Context(
        split = 'train',
        is_train = True,
        model = model,
        optimizer = optimizer,
        max_epoch = max_epoch,
        epoch_idx = start_epoch_idx,
        batch_idx = start_batch_idx,
        input = Tensor(),
        output = Tensor(),
        target = Tensor(),
        loss = Tensor(),
        metrics = dict(),
        state_dicts = [],
        logger = logger)

    class Skip(Exception): pass
    ctx.Skip = Skip

    @contextmanager
    def skip():
        try:
            yield
        except Skip:
            pass

    def run_hooks(hook_type):
        if isinstance(hooks, dict) and hook_type in hooks:
            for hook in hooks.get(hook_type):
                hook(ctx)

    @contextmanager
    def session(name):
        run_hooks('on_start_' + name)
        yield
        run_hooks('on_end_' + name)

    
        
            
    
    def process(split, data_loader, is_train):
            ctx.max_batch = len(data_loader)
            ctx.split = split
            ctx.is_train = is_train

            run_hooks('on_start_split')

            if is_train:
                model.train()
            else:
                model.eval()

            #test for accuracy
            correct = 0
            correct_2 = 0
            total = 0
            total_2 = 0
            
            active_loss = 0
            non_active_loss = 0
            active_sum = 0
            non_active_sum = 0
            error_sum = 0
            mse_total = 0
            pre = []
            tar = []
            for batch_idx, input in enumerate(progress_bar(data_loader, ascii=True, desc=split, unit='batch', leave=False)):
                if isinstance(input, (list, tuple)):
                    ctx.input = [v.to(device) if torch.is_tensor(v) else v for v in input[0]]
                elif isinstance(input, dict):
                    ctx.input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in input[0].items()}
                else:
                    ctx.input = input.to(device)

                run_hooks('on_start_batch')

                with skip(), session('batch'):
                    with torch.set_grad_enabled(ctx.is_train):
                        with skip(), session('forward'):
                            ctx.output = ctx.model(ctx.input)

                        guesses = ctx.output[0].cpu().detach()
                        pre.append(guesses)
                        
                run_hooks('on_end_batch')
            
            
            #smoothe the prediction
            pred = torch.cat(pre, 0).detach().numpy()
            frames = pred.shape[0]
            splined_data = np.zeros([frames,pred.shape[1]])
            for i in range(0,pred.shape[1]):
                b = pred[:,i]
                spline = scipy.signal.cspline1d(b, lamb=10)
                splined_data[:,i]=spline
            os.makedirs(os.path.dirname(saved_path))
            np.save(saved_path, splined_data)
            run_hooks('on_end_split')

    
    
    
    
    run_hooks('on_start')
    for split, loader in test_loaders:
        process(split, loader, False)
    run_hooks('on_end')

    return ctx
