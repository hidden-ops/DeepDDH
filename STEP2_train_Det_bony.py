import argparse
import logging
import os
import sys
import cv2

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddhnet_model.dilated_ddhnet_bony import DDHNet_Bony

from utils.eval_bony import eval_net
from utils.dataset_bony import BasicDataset
from utils.segLoss import SegmentationLosses
from utils.early_stopping import EarlyStopping
from utils.CentroidMSELoss import CentroidMSELoss

tr_dir_img = 'data/seg_detD_bony/train/imgs/'
tr_dir_mask = 'data/seg_detD_bony/train/seg/'
tr_dir_line_mark = 'data/seg_detD_bony/train/bony_mask/'

val_dir_img = 'data/seg_detD_bony/val/imgs/'
val_dir_mask = 'data/seg_detD_bony/val/seg/'
val_dir_line_mark = 'data/seg_detD_bony/val/bony_mask/'

te_dir_img = 'data/seg_detD_bony/test/imgs/'
te_dir_mask = 'data/seg_detD_bony/test/seg/'
te_dir_line_mark = 'data/seg_detD_bony/test/bony_mask/'

dir_checkpoint = './checkpoint/seg_detD_bony/'
base_path =  './checkpoint/seg_detD_bony/'
model_name = 'ddhnet'
load_path = './checkpoint/PreD/dilated_ddhnet_b_4_true/1/best_model.pth'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.0001,
              save_cp=False,
              load_cp=None,
              img_scale=0.5,
              alpha=0.5):

    train = BasicDataset(tr_dir_img, tr_dir_mask, tr_dir_line_mark, batch_size, img_scale)
    val = BasicDataset(val_dir_img, val_dir_mask, val_dir_line_mark, batch_size, img_scale)
    test = BasicDataset(te_dir_img, te_dir_mask, te_dir_line_mark, batch_size, img_scale)
    n_val = len(val)
    n_train = len(train)
    n_test = len(test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    writer = SummaryWriter(comment=f'Model_{model_name}_LR_{lr}_BS_{batch_size}')
    global_step = 0
    early_stopping = EarlyStopping(patience=10, delta=0.00001, verbose=True,
                                   checkpoint_path=os.path.join(dir_checkpoint, 'best_model.pth'))

    logging.info(f'''Starting training:
        Epochs:                {epochs}
        Batch size:            {batch_size}
        Learning rate:         {lr}
        Training size:         {n_train}
        Validation size:       {n_val}
        Test size:             {n_test}
        Checkpoints:           {save_cp}
        Images scaling:        {img_scale}
        Load checkpoint path:  {load_cp}
    ''')
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-12, momentum=0.95)
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-12)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    if net.n_classes > 1:
        #criterion = nn.CrossEntropyLoss()
        criterion = SegmentationLosses(cuda=True)
        centroid_loss = CentroidMSELoss(n_class=net.n_classes)

    else:
        criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.BCELoss()
        #criterion = dice_coeff()

    if net.bony_class > 1:
        #criterion = nn.CrossEntropyLoss()
        criterion_bony = SegmentationLosses(cuda=True)
        centroid_bony_loss = CentroidMSELoss(n_class=net.bony_class)

    else:
        criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.BCELoss()
        #criterion = dice_coeff()

    times = 0
    tot_iter = int(n_train // batch_size)
    proc_interval = int(tot_iter // 5) # 每完成20%进行一次验证
    for epoch in range(epochs):
        net.train()
        epoch_mask = True
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                true_bony = batch['bony']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)

                mask_type = torch.float64 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                bony_type = torch.float64 if net.bony_class == 1 else torch.long
                true_bony = true_bony.to(device=device, dtype=bony_type)

                if model_name == 'bisenet':
                    aux_pred0, aux_pred1, main_pred, smax_pred = net(imgs)
                    aux_loss0 = criterion.CrossEntropyLoss(aux_pred0, true_masks)+criterion.FocalLoss(aux_pred0, true_masks, gamma=2, alpha=0.5)
                    aux_loss1 = criterion.CrossEntropyLoss(aux_pred1, true_masks)+criterion.FocalLoss(aux_pred1, true_masks, gamma=2, alpha=0.5)
                    main_loss = criterion.CrossEntropyLoss(main_pred, true_masks)+criterion.FocalLoss(main_pred, true_masks, gamma=2, alpha=0.5)
                    loss = (aux_loss0 + aux_loss1 + main_loss) / 3 + centroid_loss(main_pred, true_masks)
                elif model_name == 'danet':
                    main_pred = net(imgs)
                    aux_loss = criterion.CrossEntropyLoss(main_pred[0], true_masks) + criterion.FocalLoss(main_pred[0], true_masks, gamma=2, alpha=0.5)
                    main_loss = criterion.CrossEntropyLoss(main_pred[1], true_masks) + criterion.FocalLoss(main_pred[1], true_masks, gamma=2, alpha=0.5)
                    loss = (aux_loss + main_loss) / 2  + centroid_loss(main_pred[1], true_masks)
                else:
                    masks_pred, bony_pred = net(imgs)
                    # seg
                    ce_loss = criterion.CrossEntropyLoss(masks_pred, true_masks)
                    fl_loss = criterion.FocalLoss(masks_pred, true_masks, gamma=2, alpha=0.5)
                    ct_loss = centroid_loss(masks_pred, true_masks)
                    seg_loss = ce_loss + fl_loss + 0.1 * ct_loss

                    # bony
                    bony_ce_loss = criterion_bony.CrossEntropyLoss(bony_pred, true_bony)
                    bony_fl_loss = criterion_bony.FocalLoss(bony_pred, true_bony, gamma=2, alpha=0.5)
                    # bony_ct_loss = centroid_bony_loss(bony_pred, true_bony)
                    bony_loss = bony_ce_loss + bony_fl_loss# + 0.1 * bony_ct_loss

                    loss = seg_loss + bony_loss

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % proc_interval == int(proc_interval - 1):
                    softmax = nn.Softmax(dim=1)
                    if  epoch_mask == True:
                        if model_name == 'bisenet':
                            pred_masks = torch.max(softmax(smax_pred), 1).indices
                        elif model_name == 'danet':
                            pred_masks = torch.max(softmax(main_pred[1]), 1).indices
                        else:
                            pred_masks = torch.max(softmax(masks_pred), 1).indices
                        pred_mask = pred_masks.cpu().numpy()
                        true_mask = true_masks.cpu().numpy()
                        for t in range(0):
                            pred = pred_mask[t] * 20
                            true = true_mask[t] * 20
                            cv2.imwrite(str(epoch)+str(times)+'pred.png', pred)
                            cv2.imwrite(str(epoch)+str(times)+'true.png', true)
                        epoch_mask = False

                    print('loss:', loss.item())
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, model_name, device)
                    print('Process', global_step // proc_interval + 1, 'Validation Dice Coeff:', val_score)
                    early_stopping(val_score, net)  # 调用 EarlyStopping

                    if early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            val_score = eval_net(net, val_loader, model_name, device)
            print('Validation Dice Coeff:', val_score)
            test_score = eval_net(net, test_loader, model_name, device)
            print('Test Dice Coeff:', test_score)
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    n_classes = 8
    bony_class = 7
    model = ['dilated_ddhnet']#,
    dropouts = [1.0]
    args = get_args()
    args.load = load_path

    for k in range(len(model)):
        model_name = model[k]
        for t in range(len(dropouts)):
            for s in range(len(dropouts)):
                for i in range(1, 3):
                    for sj in range(2):
                        if sj == 0:
                            dir_checkpoint = base_path + model[k] + '_b_4_true/'+str(i)+'/'
                            args.batchsize = 4
                        else:
                            dir_checkpoint = base_path + model[k] + '_b_2_true/'+ str(i) + '/'
                            args.batchsize = 2

                        if not os.path.exists(dir_checkpoint):
                            os.makedirs(dir_checkpoint)
                        print('Iteration:', i, 'model_name:', model_name, 'checkpoint_path:', dir_checkpoint, 'dropout1:', dropouts[t], 'dropout2:', dropouts[s])
                        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        logging.info(f'Using device {device}')

                        if model_name == 'unet':
                        #net = UNet_Res(n_channels=3, n_classes=8, pretrained_model=False)
                            net = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
                        elif model_name == 'segnet':
                            net = SegNet(n_channels=3, n_classes=n_classes)
                        elif model_name == 'fcn':
                            net = FCN(n_channels=3, n_classes=n_classes, pretrained_model=True)
                        elif model_name == 'aunet':
                            net = AUNet_Res(n_channels=3, n_classes=n_classes, pretrained_model=True, learned_bilinear=True)
                        elif model_name == 'danet':
                            #net = DANet(n_classes=8, n_channels=3, pretrained_model=True)
                            net = DANet(n_classes=n_classes, n_channels=3, backbone='resnet50')
                        elif model_name == 'bisenet':
                            net = BiSeNet(n_classes=n_classes, n_channels=3, pretrained_model=True)
                        elif model_name == 'scse':
                            net = SCSERes(n_classes=n_classes, n_channels=3, pretrained_model=True)
                        elif model_name == 'ddhnet':
                            net = DDHNet(n_classes=n_classes, n_channels=3, dropout1=dropouts[t], dropout2=dropouts[s], pretrained_model=True)
                        elif model_name == 'dilated_ddhnet':
                            net = DDHNet_Bony(n_classes=n_classes, bony_class=bony_class, n_channels=3, pretrained_model=True)
                        elif model_name == 'danet_ddhnet':
                            net = danet_DDNet(n_classes=n_classes, n_channels=3, backbone='resnet50')

                        logging.info(f'Network:\n'
                             f'\t{net.n_channels} input channels\n'
                             f'\t{net.n_classes} output channels (classes)\n')
                            #f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

                        if args.load:
                            # net.load_state_dict(
                            #     torch.load(args.load, map_location=device)
                            # )
                            checkpoint = torch.load(args.load)
                            model_state_dict = net.state_dict()
                            filtered_state_dict = {k: v for k, v in checkpoint.items() if
                                                   k in model_state_dict}
                            missing_keys, unexpected_keys = net.load_state_dict(filtered_state_dict, strict=False)
                            if missing_keys:
                                print("Missing keys were not found in the checkpoint:", missing_keys)
                            if unexpected_keys:
                                print("Unexpected keys were found in the checkpoint:", unexpected_keys)

                            logging.info(f'Model loaded from {args.load}')

                        net.to(device=device)

                        try:
                            train_net(net=net,
                            epochs=args.epochs,
                            batch_size=args.batchsize,
                            lr=args.lr,
                            load_cp=args.load,
                            device=device,
                            img_scale=args.scale,
                            )
                        except KeyboardInterrupt:
                            torch.save(net.state_dict(), 'INTERRUPTED.pth')
                            logging.info('Saved interrupt')
                            try:
                                sys.exit(0)
                            except SystemExit:
                                os._exit(0)