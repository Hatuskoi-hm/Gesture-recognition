import torch
import torch.nn as nn
from torch.nn import functional
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(5, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.5)
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax()
        self.loss = functional.cross_entropy

    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.act1(x)
        x = self.dropout(x)
        pred = self.act2(x)
        if y is not None:
            return self.loss(pred, y.squeeze())
        else:
            return pred


class Predict():
    def __init__(self, config, model, logger):
        super(Predict, self).__init__()

        self.config = config
        self.model = model

        model_path = 'result/20230408-2_epoch_26.pth'
        self.model.load_state_dict(torch.load(model_path))
        print('model load finished!')
        self.logger = logger

    def pred(self, predata):
        self.model.eval()
        predata = torch.FloatTensor(predata)
        # predata = predata.cuda()
        with torch.no_grad():
            pred_result = self.model(predata)
            pred_label = torch.argmax(pred_result)
        return pred_label, pred_result


Config = {
        "train_data_path": "../datas",
        "valid_data_path": "../datas",

        "batch_size": 8,
        "optimizer": "Adam",
        "learn_rate": 1e-3,
        "model_path": "result",
    }

model = MyModel(Config)

Pred = Predict(Config, model, logger)

file_list = os.listdir("datas")
for index, cur_file in enumerate(file_list[5:10]):
    if index == 3:
        start = 147
    else:
        start = 50
    cur_file = 'datas/' + cur_file
    print(cur_file)
    all_lines = []
    with open(cur_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(";")
            line = [int(item) for item in line]
            all_lines.append(line)
        for i in range(start, len(all_lines) - 9):
            input = []
            for j in range(5):
                num = 0
                for k in range(i, i + 8):
                    num += all_lines[k][j]
                input.append(num/8)
            pred_res, pred_prob = Pred.pred(input)
            print('当前手势为:', cur_file.split('.')[0][-1])
            print("当前手势预测概率分布为:", pred_prob)
            print("当前姿势预测结果为:", pred_res.item() + 1)
            print('---------------------')
            break




