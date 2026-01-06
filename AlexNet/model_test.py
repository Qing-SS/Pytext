import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet
import pandas as pd


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                             download=True)
    # noinspection PyShadowingNames
    test_dataloader = data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader


# noinspection PyShadowingNamas
# noinspection PyShadowingNames
def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 初始化参数
    test_corrects = torch.tensor(0.0).to(device)
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == "__main__":
    model = AlexNet()
    model.load_state_dict(torch.load('best_model.pth'))
    # # 利用现有的模型进行模型的测试
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    results = []

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为验证模型
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：", classes[result], "------", "真实值：", classes[label])

            results.append({
                "预测值": classes[result],
                "真实值": classes[label]
            })

    df = pd.DataFrame(results)
    save_path = "forecast.xlsx"
    df.to_excel(save_path, index=False)
    print(f"预测结果已保存到 {save_path}")

