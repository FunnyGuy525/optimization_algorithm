def return_0_1(DC_col):
    DC_col_01 = DC_col
    for i in range(len(DC_col)):
        if DC_col_01[i] >= 1:
            DC_col_01[i] = 1
        else:
                DC_col_01[i] = 0    
    return DC_col_01

def combine(DC):
    cnt_return = np.zeros((100,7))
    cnt_return[:,0:3] = DC
    cnt_return[:,3] = return_0_1(DC[:,0] + DC[:,1])
    cnt_return[:,4] = return_0_1(DC[:,0] + DC[:,2])
    cnt_return[:,5] = return_0_1(DC[:,1] + DC[:,2])
    cnt_return[:,6] = return_0_1(DC[:,0] + DC[:,1] + DC[:,2])
    return cnt_return


# 计算当前仓库存储SKU情况下派送需要包裹数
def target(order, DC):
    # DC: 100×3 表示三个仓库包含SKU
    num_of_package = 0
    # DC_combine 100×7 分别表示各仓库以及各仓库组合包含的SKU
    DC_combine = combine(DC)
    result = np.dot(order, DC_combine)
    for i in range(50000):
        index = np.where(result[i] == order_sums[i])[0][0]
        if index <= 2:
            num_of_package += 1
        elif index <= 5:
            num_of_package += 2
        else:
            num_of_package += 3
    return num_of_package

def generate_random_array(length, num_ones):
    array = np.zeros(length)
    array[:num_ones] = 1
    np.random.shuffle(array)
    return array
# 生成满足条件的向量，每个向量长度为100，其中有70个1，30个0
def generate_arrays_and_logical_or(length, num_ones):
    array1 = generate_random_array(length, num_ones)
    array2 = generate_random_array(length, num_ones)
    array3 = generate_random_array(length, num_ones)

    result_array = np.logical_or(np.logical_or(array1, array2), array3)
    
    while not np.all(result_array):
        array1 = generate_random_array(length, num_ones)
        array2 = generate_random_array(length, num_ones)
        array3 = generate_random_array(length, num_ones)
        result_array = np.logical_or(np.logical_or(array1, array2), array3)
    
    return array1, array2, array3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm#进度条设置
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

np.random.seed(2377) # 随机数种子，保证结果可重复

class BackPack(object):
    def __init__(self):
        self.N = 100  # 群体粒子个数
        self.D = 300  # 粒子维数
        self.T = 100  # 最大迭代次数
        self.c1 = 1.5  # 学习因子1
        self.c2 = 1.5  # 学习因子2
        self.w = 1  # 惯性因子，一般取1
        self.V_max = 10  # 速度最大值
        self.V_min = -10  # 速度最小值
        self.G=100 #迭代次数

    #初始化种群
    def init_x(self):

        X = []
        for _ in range(self.N):
            X.append(np.concatenate(generate_arrays_and_logical_or(100, 70)))
            
        X = np.array(X)
        return X

    #初始化速度
    def init_v(self):
        """
        :return: 速度
        """
        V = np.random.random(size=(self.N, self.D))  ##10维
        return V


    # 适应度值
    def fitness_func(self,X):

        size = np.shape(X)[0] #种群个数
        result = np.zeros((size, 1))
        for i in range(size):  # 遍历每一个粒子
            DC = X[i].reshape(3, 100).T
            
            result[i] = target(order, DC)
        return result  # 我们要求result越大

    # 速度更新公式
    def velocity_update(self,V, X, pbest, gbest):
        """
        根据速度更新公式更新每个粒子的速度
         种群size=100
        :param V: 粒子当前的速度矩阵，size*10 的矩阵
        :param X: 粒子当前的位置矩阵，size*10 的矩阵
        :param pbest: 每个粒子历史最优位置，size*10 的矩阵
        :param gbest: 种群历史最优位置，1*10 的矩阵
        """
        r1 = np.random.random((self.N, 1)) #(粒子个数,1)
        r2 = np.random.random((self.N, 1))
        V = self.w * V + self.c1 * r1 * (X - pbest) + self.c2 * r2 * (X - gbest)  # pbest-X改为X-pbest
        # 防止越界处理
        V[V < self.V_min] = np.random.random() * (self.V_max - self.V_min) + self.V_min
        V[V > self.V_max] = np.random.random() * (self.V_max - self.V_min) +self.V_min
        return V

    # 位置更新公式
    def position_update(self,X, V):
        """
        根据公式更新粒子的位置
        :param X: 粒子当前的位置矩阵，维度是 size*10
        :param V: 粒子当前的速度举着，维度是 size*10
        return  更新后的种群
      """

        for i in range(self.N):  # 遍历每一个粒子
            # 修改速度为sigmoid形式

            V[i, :] = 1. / (1 + np.exp(-np.array(V[i, :])))
            for j in range(self.D):  # 遍历粒子中的每一个元素
                rand = np.random.random()  # 生成 0-1之间的随机数
                if V[i, j] > rand:
                    if X[i, j] == 0:
                        for k in range(100):
                            if j//100 == 0:
                                if X[i, k] == 1 and V[i, k] < rand and (X[i, 100+k] or X[i,200+k]):
                                    X[i, 100 + k] = 0
                            if j//100 == 1:
                                if X[i, k] == 1 and V[i, k] < rand and (X[i, k-100] or X[i,100+k]):
                                    X[i, 100 + k] = 0
                            if j//100 == 2:
                                if X[i, k] == 1 and V[i, k] < rand and (X[i, k-100] or X[i,k-200]):
                                    X[i, 100 + k] = 0
                        X[i, j] = 1
                else:
                    if X[i, j] == 1:
                        for k in range(100):
                            if j//100 == 0:
                                if X[i, k] == 1 and V[i, k] > rand and (X[i, 100+k] or X[i,200+k]):
                                    X[i, 100 + k] = 1
                            if j//100 == 1:
                                if X[i, k+100] == 1 and V[i, k+100] > rand and (X[i, k] or X[i,200+k]):
                                    X[i, 100 + k] = 1
                            if j//100 == 2:
                                if X[i, k+200] == 1 and V[i, k+200] > rand and (X[i, k] or X[i,k+100]):
                                    X[i, 100 + k] = 1
                    X[i, j] = 0
            #对当前个体进行限制
            while np.sum(X[i][:100]) > 70 or np.sum(X[i][100:200]) > 70 or np.sum(X[i][200:]) > 70 or 0 in X[i][:100]+X[i][100:200]+X[i][200:] :#如果当前粒子超重
                   X[i] = np.concatenate(generate_arrays_and_logical_or(100, 70))
            
        return X

    def update_pbest(self,X, fitness, pbest, pbestfitness, m):
        """
        更新个体最优
        :param X: 当前种群
        :param fitness: 当前每个粒子的适应度
        :param pbest: 更新前的个体最优解
        :param pbestfitness: 更新前的个体最优适应度
        :param m: 粒子数量
        :return: 更新后的个体最优解、个体最优适应度
        """

        for i in range(m):
            if fitness[i] < pbestfitness[i]: # 大于改小于，越小越好
                pbest[i] = X[i]
                pbestfitness[i] = fitness[i]
        return pbest, pbestfitness

    def update_gbest(self,pbest, pbestfitness, gbest, gbestfitness, m):
        """
        更新全局最优解
        :param pbest: 粒子群
        :param pbestfitness: 个体适应度(个体最优)
        :param gbest: 全局最优解
        :param gbestfitness: 全局最优解
        :param m: 粒子数量
        :return: gbest全局最优，g对应的全局最优解
        """
        for i in range(m):
            if pbestfitness[i] < gbestfitness: # 大于改小于
                gbest = pbest[i]
                gbestfitness = pbestfitness[i]
        return gbest, gbestfitness


    def main(self):
        fitneess_value_list = []  # 记录每次迭代过程中的种群适应度值变化
        x = self.init_x()  # 初始化x
        v = self.init_v()  # 初始化v
        # 计算种群各个粒子的初始适应度值
        p_fitness = self.fitness_func(x)
        # 计算种群的初始最优适应度值
        g_fitness = p_fitness.max()
        # 讲添加到记录中
        fitneess_value_list.append(g_fitness)
        # 初始的个体最优位置和种群最优位置
        pbest = x
        gbest = x[p_fitness.argmax()]  #

        # 接下来就是不断迭代了
        for i in tqdm(range(self.G)):
            pbest=pbest.copy()#必须加，不然会被篡改值，造成结果错
            p_fitness= p_fitness.copy()#必须加，不然会被篡改值，造成结果错
            gbest=gbest.copy()#必须加，不然会被篡改值，造成结果错
            g_fitness=g_fitness.copy()#必须加，不然会被篡改值，造成结果错
            v = self.velocity_update(v, x, pbest, gbest)  # 更新速度
            x = self.position_update(x, v)  # 更新位置
            p_fitness2 = self.fitness_func(x)  # 计算子代各个粒子的适应度

            # 更新每个粒子的历史最优位置
            pbest, p_fitness=self.update_pbest(x, p_fitness2, pbest, p_fitness, self.N)

            #更新群体的最优位置
            gbest, g_fitness=self.update_gbest(pbest, p_fitness, gbest, g_fitness, self.N)

            # 记录最优迭代结果
            fitneess_value_list.append(g_fitness)

        print("最优适应度是：%.5f" % fitneess_value_list[-1])
        print("最优解是", gbest)


        plt.plot(fitneess_value_list,label='迭代曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度')
        plt.legend()
        plt.show()

if __name__=='__main__':
    pso = BackPack()
    pso.main()
