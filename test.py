import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from model import UNET
import matplotlib.pyplot as plt
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    SNR,
)
import copy


def Test(BATCH_SIZE):

    VAL_IMG_DIR = "DataPatches/AVOBigValidation/"
    VAL_TARGET_DIR = "DataPatches/AVOBigValidation/"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("RepoTest.pth.tar"), model)




    _, val_loader = get_loaders(
        None,
        None,
        VAL_IMG_DIR,
        VAL_TARGET_DIR,
        BATCH_SIZE,
        None,
        None,
        2,
        True,
    )

    SNR(val_loader, model, device=DEVICE)


def Single_Image(Image_Name, factor):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("70Epochs_reflect_dialation.pth.tar"), model)


    Target = np.loadtxt(Image_Name, delimiter=',')
    # Target = ImageOps.grayscale(Target)

    

    test = copy.deepcopy(Target)

    rr = np.arange(0, 79)
    rando = np.random.choice(rr, size=(1, int(factor * 80)), replace=False)
    test[:, rando] = 0

    test_tensor   = torch.from_numpy(test)
    test_tensor = test_tensor.to(DEVICE).float().unsqueeze(0).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        preds =  model(test_tensor)
    preds = preds.cpu().detach().numpy()
    img = preds[0, 0, :, :]

    plt.subplot(1, 3, 1)
    plt.imshow(Target, 'seismic')
    plt.title('Actual', fontsize=8)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(test, 'seismic')
    plt.title('Corrupted', fontsize=8)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img, 'seismic')
    plt.title('Reconstructed', fontsize=8)
    plt.axis('off')


    plt.show()
    


if __name__ == '__main__':
    Test()

