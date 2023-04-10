import numpy as np

font_s = 16
for index, name in enumerate(['XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX']):
    # 训练集
    datas = []
    labels = []
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
                    input.append(num/8)
                label = all_lines[i][-1] - 1
                datas.append(input)
                labels.append(label)

    datas = np.array(datas)

    # 2D图片
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(datas)
    datas_pca = pca.transform(datas)

    x1 = datas_pca[:, 0]
    x2 = datas_pca[:, 1]

    labels = np.array(labels)

    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

    ax = plt.figure().add_subplot()
    ax.scatter(x1, x2, c=labels, cmap='viridis')
    ax.set_xlabel('特征1', fontdict={"family": "SimHei", "size": font_s})
    ax.set_ylabel('特征2', fontdict={"family": "SimHei", "size": font_s})
    ax.set_title(f"测试者{index+1}训练集可视化", fontdict={"family": "SimHei", "size": font_s})
    # ax.legend()
    pname = "PCA_R/" + '2D训练集' + name + '.png'
    plt.savefig(pname, dpi=1200, bbox_inches='tight')

    plt.show()


    # 测试集
    datas = []
    labels = []
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
                    input.append(num/8)
                label = all_lines[i][-1] - 1
                datas.append(input)
                labels.append(label)

    datas = np.array(datas)

    # 2D图片
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(datas)
    datas_pca = pca.transform(datas)

    x1 = datas_pca[:, 0]
    x2 = datas_pca[:, 1]

    labels = np.array(labels)

    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

    ax = plt.figure().add_subplot()
    ax.scatter(x1, x2, c=labels, cmap='viridis')
    ax.set_xlabel('特征1', fontdict={"family": "SimHei", "size": font_s})
    ax.set_ylabel('特征2', fontdict={"family": "SimHei", "size": font_s})
    ax.set_title(f"测试者{index+1}测试集可视化", fontdict={"family": "SimHei", "size": font_s})
    # ax.legend()
    pname = "PCA_R/" + '2D测试集' + name + '.png'
    plt.savefig(pname, dpi=1200, bbox_inches='tight')

    plt.show()
