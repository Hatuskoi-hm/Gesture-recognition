import numpy as np


datas = []
labels = []
for index, name in enumerate(['XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX']):
    # 训练集
    file_list = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt']
    for cur_file in file_list[:5]:
        cur_file = 'datas/' + name + cur_file
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
                    input.append(num / 8)
                label = all_lines[i][-1] - 1
                datas.append(input)
                labels.append(label)

datas = np.array(datas)

# 3D图片
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(datas)
datas_pca = pca.transform(datas)

x1 = datas_pca[:, 0]
x2 = datas_pca[:, 1]
x3 = datas_pca[:, 2]

labels = np.array(labels)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x1, x2, x3, c=labels)
ax.set(xlabel='特征1', ylabel='特征2', zlabel='特征3')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
# 保存图片
pname = "PCA_R/" + '3D训练集' + "整体" + '.png'
plt.savefig(pname, dpi=1200, bbox_inches='tight')
plt.show()

    # 测试集
datas = []
labels = []
for index, name in enumerate(['XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX']):
    file_list = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt']
    for cur_file in file_list[:5]:
        cur_file = 'validdatas/' + name + cur_file
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
                    input.append(num / 8)
                label = all_lines[i][-1] - 1
                datas.append(input)
                labels.append(label)

datas = np.array(datas)

# 3D图片
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(datas)
datas_pca = pca.transform(datas)

x1 = datas_pca[:, 0]
x2 = datas_pca[:, 1]
x3 = datas_pca[:, 2]

labels = np.array(labels)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x1, x2, x3, c=labels)
ax.set(xlabel='特征1', ylabel='特征2', zlabel='特征3')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
# 保存图片
pname = "PCA_R/" + '3D测试集' + "整体" + '.png'
plt.savefig(pname, dpi=1200, bbox_inches='tight')
plt.show()
