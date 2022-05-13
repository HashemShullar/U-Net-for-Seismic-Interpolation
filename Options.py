import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('--patch_name', type=str, default='gather373_i13_j1.txt', help="Specify image name. Only .jpg images can be used")
    parser.add_argument('--lr', type=int, default=1e-4, help="Learning rate used by the optimizor")
    parser.add_argument('--batch_size', type=int, default=16, help="Numbe of images in each iteration")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--mode', type=str, default='train', help="'train', 'test' or 'interpolate'. 'interpolate' is for testing on a single image")
    parser.add_argument('--corruption_precentage', type=float, default=0.5, help=" % of traces to be removed from the image (For mode (interpolate))")

    args = parser.parse_args()
    return args

