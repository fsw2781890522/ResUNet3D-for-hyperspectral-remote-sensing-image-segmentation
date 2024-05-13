import torch
import torchvision
from torchsummary import summary
from torch import nn
from model2D import ResUNet2D
from model3D import ResUNet3D


def calculate_output_shape(model, input_shape, batch_size=1):
    # 创建一个模拟的输入张量
    input_tensor = torch.randn(batch_size, *input_shape)
    # 设置模型为评估模式
    model.eval()
    # 使用模型进行前向传播
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # 返回输出张量的形状
    return output_tensor.shape


# net = torchvision.models.segmentation.DeepLabV3()

# vgg = torchvision.models.vgg16(pretrained=False)
# vgg.features[0] = nn.Conv2d(10, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# vgg.classifier[6] = nn.Linear(4096, 10)
# print(vgg)

# net = torchvision.models.segmentation.fcn_resnet50()
# net = torchvision.models.resnet152(pretrained=False)
# net = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
# """
# (4): Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))
# """
# net.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))


if __name__ == '__main__':
    # net = torchvision.models.resnet18(pretrained=False)
    # net = ResUNet3D(num_bands=15, num_classes=20)
    # net = torchvision.models.resnet18(pretrained=False)
    # print(net)

    net = ResUNet3D(188, 12)
    print(net)

    summary(model=net.to('cuda:0'), input_size=(1, 188, 16, 16), batch_size=10)
