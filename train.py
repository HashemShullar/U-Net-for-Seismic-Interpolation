import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    SNR,
)

import copy



LOSS           = []
AVGSNR         = []



LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 16
NUM_EPOCHS    = 10
NUM_WORKERS   = 2
PIN_MEMORY    = True
LOAD_MODEL    = False


TRAIN_IMG_DIR    = 'C:/Users/hashe/Desktop/KFUPM/212/EE 562/DSP1_TermProject/Training_Patches_Targets_AVO'
TRAIN_TARGET_DIR = 'C:/Users/hashe/Desktop/KFUPM/212/EE 562/DSP1_TermProject/Training_Patches_Targets_AVO'
VAL_IMG_DIR      = "DataPatches/AVOBigValidation/"
VAL_TARGET_DIR   = "DataPatches/AVOBigValidation/"




def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    adderloss = []
    for batch_idx, (data, targets) in enumerate(loop):


        data    = data.float().to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)


        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)



            loss = loss_fn(predictions, targets)
            adderloss.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # importance.step()
        scaler.update()

            # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return sum(adderloss)/len(adderloss)

def main(): 
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ],
    )





    model    = UNET(in_channels=1, out_channels=1).to(DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("60Epochs_reflect_dialation.pth.tar"), model)

    loss_fn = nn.L1Loss(reduction='mean')


    optimizer  = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_TARGET_DIR,
        VAL_IMG_DIR,
        VAL_TARGET_DIR,
        BATCH_SIZE,
        train_transform,
        None,
        NUM_WORKERS,
        PIN_MEMORY,
    )




    # temp_snr = SNR(val_loader, model, device=DEVICE)
    # best_snr = temp_snr
    # best_model_wts = copy.deepcopy(model.state_dict())
    # optim_cor      = copy.deepcopy(optimizer.state_dict())

    # SNR(val_loader_oldTask, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(epoch+1)
        losss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        LOSS.append(losss)

        # temp_snr = SNR(val_loader, model, device=DEVICE)

        # save model
        if 1: # temp_snr > best_snr:
            best_model_wts = copy.deepcopy(model.state_dict())
            optim_cor = copy.deepcopy(optimizer.state_dict())
            best_snr = temp_snr

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)


        AVGSNR.append(temp_snr)




if __name__ == "__main__":
    main()