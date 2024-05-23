from transformers import AdamW
import torch

# EPOCH = 0
# LR = 0
# device = torch.device("cuda")


def run(model, traindataloader):
    loss = torch.nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(params=model.parameters(), lr=LR, eps=1e-8)

    for it in range(EPOCH):
        print("Epoch：", it)
        for pos in traindataloader:
            x = pos[0].to(device)
            y = pos[1].to(device).unsqueeze(0)
            label = torch.tensor([pos[2]]).long().to(device)
            output = model(x, y).to(device)

            # 计算损失值
            los = loss(output, label)
            optimizer.zero_grad()
            los.backward()
            optimizer.step()

        with torch.no_grad():
            test(model)

    return model


def test(model, testdataloader):
    Ping = ConfusionMatrix(7, list(range(7)))
    true_ = []
    pred_ = []
    for pos in testdataloader:
        x = pos[0].to(device)
        y = pos[1].to(device).unsqueeze(0)
        label = torch.tensor([pos[2]]).long().to(device)
        output = model(x, y).to(device)

        output = torch.nn.Softmax(dim=1)(output)
        pred = torch.max(output, 1)[1]

        true_.append(label.item())
        pred_.append(pred.item())
    Ping.update(pred_, true_)
    Ping.summary()