import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

torch.manual_seed(21)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        x = self.layer3(x)
        x = self.act1(x)
        pred = self.act2(x)
        if y is not None:
            # print("pred, y.squeeze()", pred.shape, y.shape)
            return self.loss(pred, y.squeeze(-1))
        else:
            return pred


class DataGenerator:
    def __init__(self, config, data_path, flag=True):
        super(DataGenerator, self).__init__()
        self.config = config
        self.flag = flag
        self.data_path = data_path
        self.data = []
        self.load()
        print(len(self.data))

    def load(self):
        file_list = os.listdir(self.data_path)
        # print(file_list[5:])
        if self.flag:
            # file_list = file_list[:]
            file_list = file_list[:5] + file_list[10:]
            # file_list = file_list[40:]
            print("train file:", file_list)
            for cur_file in file_list:
                cur_file = self.data_path + '/' + cur_file
                all_lines = []
                with open(cur_file, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip().split(";")
                        line = [int(item) for item in line]
                        all_lines.append(line)
                    for i in range(len(all_lines) - 9):
                        input = []
                        for j in range(5):
                            num = 0
                            for k in range(i, i + 8):
                                num += all_lines[k][j]
                            input.append(num/8)
                        label = all_lines[i][-1] - 1
                        self.data.append([torch.FloatTensor(input),
                                         torch.LongTensor([label])])
        else:
            file_list = file_list[5:10]
            print("test file:", file_list)
            for cur_file in file_list:
                cur_file = self.data_path + '/' + cur_file
                all_lines = []
                with open(cur_file, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip().split(";")
                        line = [int(item) for item in line]
                        all_lines.append(line)
                    for i in range(len(all_lines) - 9):
                        input = []
                        for j in range(5):
                            num = 0
                            for k in range(i, i + 8):
                                num += all_lines[k][j]
                            input.append(num/8)
                        label = all_lines[i][-1] - 1
                        self.data.append([torch.FloatTensor(input),
                                         torch.LongTensor([label])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True, flag=True):
    dg = DataGenerator(config, data_path, flag=flag)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


class Evaluate():
    def __init__(self, config, model, logger):
        super(Evaluate, self).__init__()
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False, flag=False)

    def eval(self, epoch):
        self.model.eval()
        self.logger.info("第%d轮模型测试！" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            batch_data = [d.cuda() for d in batch_data]
            input, label = batch_data
            with torch.no_grad():
                pred_result = self.model(input)
            self.write_states(label, pred_result)
        acc = self.show_states()
        return acc

    def write_states(self, label, pred_result):
        assert len(label) == len(pred_result)
        for true_label, pred_label in zip(label, pred_result):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_states(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)


def ChooseOptimizer(config, model):
    optimizer = config["optimizer"]
    learn_rate = config["learn_rate"]
    if optimizer == "Adam":
        return Adam(model.parameters(), lr=learn_rate)
    elif optimizer == "SGD":
        return SGD(model.parameters(), lr=learn_rate)


def train(config):
    train_data = load_data(config["train_data_path"], config, shuffle=True, flag=True)
    model = MyModel(config)
    optimizer = ChooseOptimizer(config, model)
    model = model.cuda()
    evaluator = Evaluate(config, model, logger)

    accs = []
    epoch_loss = []
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            inputs, labels = batch_data
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(float(loss))

            if index % int(len(train_data) / 1) == 0:  # print twice every epoch
                logger.info("epoch: %d; batch_loss %f" % (epoch, loss))
        epochloss = np.mean(train_loss)
        epoch_loss.append(epochloss)
        if epoch % 2 == 0:
            model_path = os.path.join(config["model_path"], "20230408-2_epoch_%d.pth" % epoch)
            torch.save(model.state_dict(), model_path)
        model_path = os.path.join(config["model_path"], "20230408-2_epoch_%d.pth" % config["epochs"])
        torch.save(model.state_dict(), model_path)
        acc = evaluator.eval(epoch)
        accs.append(acc)
    return accs, epoch_loss


if __name__ == "__main__":
    Config = {
        "train_data_path": "datas",
        "valid_data_path": "validdatas",

        "batch_size": 8,
        "optimizer": "Adam",
        "learn_rate": 1e-3,

        "model_path": "result",
        "epochs": 100,
    }
    accs, epoch_loss = train(Config)
    x = [i+1 for i in range(Config['epochs'])]
    plt.plot(x, accs, 'red', label="acc")
    plt.legend()

    pname = "result1/" + "1acc" + '.png'
    plt.savefig(pname, dpi=1200, bbox_inches='tight')

    plt.show()

    x = [i + 1 for i in range(Config['epochs'])]
    plt.plot(x, epoch_loss, 'red', label="train_loss")
    plt.legend()

    pname = "result1/" + "1loss" + '.png'
    plt.savefig(pname, dpi=1200, bbox_inches='tight')

    plt.show()
