import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from my_dataset import DriveDataset

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 4  # exclude background
    weights_path = "/content/drive/MyDrive/save_weights_thesis/best_model.pth"
    img_path = "/content/unet/Dataset_PDB/test/CMAP/1PZ7_A_origin.png"
    #roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    #assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

        # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    confmat, dice = evaluate(model, val_loader, device=device, num_classes=classes+1)
    val_info = str(confmat)
    print(val_info)
    print(f"dice coefficient: {dice:.3f}")

    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        
        color = [[0,0,0],[255,0,0], [0,255,0], [0,0,255], [255,255,255]]
        # red: helix, green: sheet, blue: anti-parallel, white: parallel
        color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype='uint8')
        for i in range(0, 5):
            color_mask[np.where(prediction==i)] = color[i]
        #prediction[prediction == 0] = 0
        #prediction[prediction == 1] = 255
        #prediction[prediction == 2] = 100
        #prediction[prediction == 3] = 200
        #prediction[prediction == 4] = 150
        # 将不敢兴趣的区域像素设置成0(黑色)
        #prediction[roi_img == 0] = 0
        mask = Image.fromarray(color_mask)
        mask.save("test_result.png")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DATASET root")
    # exclude background
    parser.add_argument("--num-classes", default=4, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=5, type=int)
    parser.add_argument("-p", "--partition-idx", default=0, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
