import sys
sys.path.insert(0, '..')
import os
import re
import argparse
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from model.resfcn import ResFCN
from train.dataset import WholeDataset
from train.schedulers import PolyLR
from apex import amp
from common.utils import Progbar


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='None')
    parser.add_argument('--name', type=str, default='test')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse()


    train_set = WholeDataset(7000)
    val_set   = WholeDataset(1000)

    model = ResFCN()
    loss_func = nn.CrossEntropyLoss()

    optimizer = Adam(params=model.parameters(),
                     lr=1e-3,
                     weight_decay=5e-4)

    scheduler = PolyLR(optimizer=optimizer)

    writer_dir = './write'
    save_dir = './ckpt'
    os.makedirs(writer_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(logdir=writer_dir)

    board_num = 0
    start_epoch = 0



    train_data_loader = DataLoader(
        train_set,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    val_data_loader = DataLoader(
        train_set,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True)


    if os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()})
        model.train(mode=True)
        start_epoch = checkpoint['epoch'] + 1
        board_num = 1
        print("load %s finish" % opt.resume)


    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.cuda()
    if opt.fp16:
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale='dynamic')


    model.train()

    best_val_loss = 10000

    for epoch in range(start_epoch, 10):
        # train
        progbar = Progbar(len(train_data_loader), stateful_metrics=['epoch','type'])

        model.train()
        for data in train_data_loader:
            optimizer.zero_grad()
            img, lab = data
            img = img.cuda()
            lab = lab.cuda()

            pred = model(img)
            loss = loss_func(pred, lab)
            if opt.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            progbar.add(1, values=[('epoch', epoch),
                                   ('type','train'),
                                   ('train_loss', loss.item()),])
        # validation
        model.eval()
        val_losses = 0
        progbar = Progbar(len(val_data_loader), stateful_metrics=['epoch','type'])
        with torch.no_grad():
            for data in val_data_loader:
                img, lab = data
                img = img.cuda()
                lab = lab.cuda()
                pred = model(img)
                val_loss = loss_func(pred, lab)
                progbar.add(1, values=[('epoch', epoch),
                                       ('type', 'val'),
                                       ('train_loss', val_loss.item()), ])
                val_losses += val_loss.item()
        val_losses /= len(val_data_loader)
        print("epoch: %d, avg_val_loss: %.4f"%(epoch, val_losses))

        torch.save({
            'epoch': epoch,
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, 'model_%d.pt' % epoch))
        best_val_loss = min(val_loss, best_val_loss)
        writer.add_scalars(opt.name, {"val_loss": val_losses, "bst_val_loss": best_val_loss}, epoch)

