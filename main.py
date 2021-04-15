import torch.nn as tn
import torch.optim
import model.ecg_model as em
import utils.dataloader as edl
import torch as th


model = em.cnn_ecg_model()

def train():
    criterion = tn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_loader=edl.fun()
    for epoch in range(5):
        print('第%d轮学习.......'%epoch)
        for i,data in enumerate(train_loader,0):
            inputs,target=data
            #inputs=th.unsqueeze(inputs,1)
            inputs=inputs.float()
            target=target.long()
            inputs=inputs.permute(0,2,1)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("损失率 %f"%)


def test():
    correct=0
    total=0
    test_loader=edl.funTest()
    print('开始测试.......')
    with th.no_grad():
        for data in test_loader:
            inputs,target=data
            inputs=inputs.float()
            inputs=inputs.permute(0,2,1)
            target=target.long()
            outputs=model(inputs)
            _,predicted=th.max(outputs.data,dim=1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
            rate=correct/total
    print('rate is %f'%rate)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train()
    test()
