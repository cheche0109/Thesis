import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        # 在torch中默认将索引放在1的位置，所以return中用permute的方法将chanel纬度的数据给移到索引为1的位置
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient

    # x是针对某一个类别的预测概率矩阵，target是针对某一个类别的GT
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size): # 遍历每一张图片
        x_i = x[i].reshape(-1) # 去除第i张图片，变成向量的形式
        t_i = target[i].reshape(-1)# 取出对应图片的target，然后变成向量形式
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask] # 找到图片中感兴趣的区域
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i) # 相应元素相乘然后求和的操作
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0: # 预测的和GT都为0，所以预测的都是对的
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)# epsilon防止出现分母为0的情况

    return d / batch_size # 得到针对每张图片的的某个类别的dice的均值


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        # x每个类别的预测值，和GT（target），然后去计算dice coefficeinet
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1] # 处以通道数：类别个数，得到所有类别的dice均值


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    # 对预测的数值在chanel的方向上去做softmax处理，就能得到每个像素针对每个类别的概率
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)
