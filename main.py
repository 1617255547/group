import numpy as np
import matplotlib.pyplot as plt



#设置成load data函数,方便调用
def load_data():
    datafile = 'C:/Users/86180/Downloads/housing.data'             #传文件数据进去
    data = np.fromfile(datafile, sep=' ')
    print(data)
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MED']
    feature_num = len(feature_names)                    #每条数据包括14项，前13是影响因素，最后一项是房屋价格中位数


    data = data.reshape([data.shape[0] // feature_num, feature_num])                   #将原始数据变成【N,14】这样的形状

    ratio = 0.8                             #将数据集拆分为训练集和测试集，二者必须无交集
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    maximums, minimums, avgs = [training_data.max(axis=0), training_data.min(axis=0), training_data.sum(axis=0) / training_data.shape[0]]  #计算训练集的最大，最小，平均值

    global max_values                       #记录数据的归一化参数，在预测时对数据 做归一化
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    for i in range(feature_num):            #对数据做归一化处理
        data[:, 1] = (data[:, i]-minimums[i]) / (maximums[i] - minimums[i])

    training_data = data[:offset]           #训练集和测试集的划分比例
    test_data = data[offset:]
    return training_data, test_data

class Network(object):
    def __init__(self, num_of_weights):
        self.w = np.random.randn(num_of_weights, 1)  #随记产生w的初始值
        self.b = 0.
    def forward(self, x):
        z = np.dot(x, self.w) + self.b           #预测的房价
        return z

    def loss(self, z, y):                         #设立损失函数，输出均方差
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        np.seterr(over='ignore', invalid='ignore')
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):                     #求方向导数最大值，即求梯度并返回
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z - y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z - y)
        return gradient_w, gradient_b
    def update(self, gradient_w, gradient_b, eta=0.01):            #利用梯度下降法，设置学习率为0.01
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses  = []
        for epoch_id in range(num_epochs):                      #在每次迭代之前，将训练数据打乱，再按每次取规定数的数据取出
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]            #数据拆分打包，每个mini_batch单位中包含batch_size条的数据
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:4f}'.
                      format(epoch_id, iter_id, loss))

        return losses


train_data, test_data = load_data()          #获取数据集
net = Network(13)                           #创建网络
losses = net.train(train_data, num_epochs=50, batch_size=10, eta=0.01)     #开始训练

plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()                                  #画出损失函数的变化趋势


def load_one_example():
    idx = -10
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    one_data = one_data.reshape([1, -1])            #修改该条数据的shape为【1,13】

    return one_data, label
one_data, lable = load_one_example()
predict = net.forward(one_data)
predict = predict * (max_values[-1] - min_values[-1]) + min_values[-1]
print(f'Inference result is {predict}, the corresponding lable is {lable}')



