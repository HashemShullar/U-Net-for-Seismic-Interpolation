from Options import args_parser
from train import Train
from test import Test, Single_Image


if __name__ == "__main__":
    args = args_parser()
    mode = args.mode

    Image_Name    = args.patch_name
    LEARNING_RATE = args.lr
    BATCH_SIZE    = args.batch_size
    NUM_EPOCHS    = args.epochs
    factor        = args.corruption_precentage
    
    if mode == 'train':
        Train(LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
        
    elif mode == 'test':
        Test(BATCH_SIZE)
    
    else:
        Single_Image(Image_Name, factor)
    

    






















