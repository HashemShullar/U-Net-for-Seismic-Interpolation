import torch
import torchvision
from dataset import Seismic
from torch.utils.data import DataLoader
import numpy as np


def save_checkpoint(state, filename="NewModel.pth.tar"):
    print("---> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("---> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_targetdir,
    val_dir,
    val_targetdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Seismic(
        image_dir=train_dir,
        target_dir=train_targetdir,
        transform=train_transform,
        validation_flag=0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,

    )

    val_ds = Seismic(
        image_dir=val_dir,
        target_dir=val_targetdir,
        transform=val_transform,
        validation_flag = 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def SNR(loader, model, device="cuda"):
    model.eval()
    snr = []
    with torch.no_grad():
        for x, y in loader:


            x = x.to(device).float().unsqueeze(1)
            y = y.to(device).unsqueeze(1)

            
            preds =  model(x)

            snr_temp = 0
            for item in range(x.shape[0]):
                snr_temp += 10*torch.log10(((y[item, :, :, :]**2).sum())/( ( ( y[item, :, :, :] - preds[item, :, :, :] )**2 ).sum() ))


            snr.append(snr_temp/16)

    print(f"Got SNR of: {sum(snr)/len(snr):.2f}")
    model.train()
    return sum(snr)/len(snr)




# def TasksData(LocList, LocListTargets, transform, NumOfTasks):
#     DataList = []
#     for i in range(NumOfTasks):
# 
#         ds = Seismic(
#             image_dir=LocList[i],
#             target_dir=LocListTargets[i],
#             transform=transform,
#             validation_flag=1,
#         )
# 
#         DataList.append(ds)
# 
#     return DataList


    
    
    
