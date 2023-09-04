for i in tqdm(range(n)):
    print('第', i + 1, '代')
    pop1 = l_pop;  # 存下以调用
    # 计算适应度
    fit = np.zeros(num)
    for i in range(num):
        DC = l_pop[i].reshape(3, 100).T
        fit[i] = target(order, DC)
    fit = 1e5 - fit
    fit1 = (fit - np.min(fit) + 1) * 3;
    prob = fit1 / (np.sum(fit1))
    for i in range(num):
        probability = np.random.rand()  # 生成随机数  
        s = 0
        for j in range(num):
            s += prob[j]
            if s > probability:
                count = j
                break
        n_pop[i, :] = l_pop[count, :]

    n_pop = l_pop.copy()
    all_index = np.random.permutation(num)
    for i in range(1, num // 2 + 1):
        ind = all_index[2 * i: 2 * i + 2]
        probability = 0.5  # 交叉概率

        if np.random.rand() < probability:
            pos = np.random.randint(1, dim // 3 + 1)  # 选择交叉的位置
            for j in range(3):
                n_pop[ind[0], pos + 100 * j:100 * (j + 1)] = l_pop[ind[1], pos + 100 * j:100 * (j + 1)]
                n_pop[ind[1], pos + 100 * j:100 * (j + 1)] = l_pop[ind[0], pos + 100 * j:100 * (j + 1)]

                # 检查新生成的ind(1)和ind(2)行是否符合70个的限制
                flag1 = np.sum(n_pop[ind[0], 1 + 100 * j:100 * (j + 1)])
                flag1 = int(flag1)
                if flag1 > 70:
                    index = np.where(n_pop[ind[0], pos + 100 * j:100 * (j + 1)] == 1)[0]
                    id = np.random.choice(index, flag1 - 70, replace=False)
                    n_pop[ind[0], 100 * j + id + pos - 1] = 0
                elif flag1 < 70:
                    index = np.where(n_pop[ind[0], pos + 100 * j:100 * (j + 1)] == 0)[0]
                    id = np.random.choice(index, 70 - flag1, replace=False)
                    n_pop[ind[0], 100 * j + id + pos - 1] = 1

                flag2 = np.sum(n_pop[ind[1], 1 + 100 * j:100 * (j + 1)])
                flag2 = int(flag2)
                if flag2 > 70:
                    index = np.where(n_pop[ind[1], pos + 100 * j:100 * (j + 1)] == 1)[0]
                    id = np.random.choice(index, flag2 - 70, replace=False)
                    n_pop[ind[1], 100 * j + id + pos - 1] = 0
                elif flag2 < 70:
                    index = np.where(n_pop[ind[1], pos + 100 * j:100 * (j + 1)] == 0)[0]
                    id = np.random.choice(index, 70 - flag2, replace=False)
                    n_pop[ind[1], 100 * j + id + pos - 1] = 1

            positions1 = np.zeros((3, 70))
            ran1 = n_pop[ind[0], :]
            positions1[0, :70] = np.where(ran1[0:100] == 1)[0][:70]
            positions1[1, :70] = np.where(ran1[100:200] == 1)[0][:70]
            positions1[2, :70] = np.where(ran1[200:300] == 1)[0][:70]
            U1 = np.union1d(positions1[0, :], np.union1d(positions1[1, :], positions1[2, :]))
            flag3 = np.setdiff1d(np.arange(1, 100), U1)

            if len(flag3) > 0:
                t2 = np.intersect1d(positions1[2, :], np.union1d(positions1[0, :], positions1[1, :]))
                id1 = np.random.choice(t2, len(flag3), replace=False)
                index3 = np.union1d(np.setdiff1d(positions1[2, :], t2[id1]), flag3)
                n_pop[ind[0], 200:300] = 0
                n_pop[ind[0], 200 + index3] = 1

            positions2 = np.zeros((3, 70))
            ran2 = n_pop[ind[1], :]
            positions2[0, :70] = np.where(ran2[0:100] == 1)[0][:70]
            positions2[1, :70] = np.where(ran2[100:200] == 1)[0][:70]
            positions2[2, :70] = np.where(ran2[200:300] == 1)[0][:70]
            U2 = np.union1d(positions2[0, :], np.union1d(positions2[1, :], positions2[2, :]))
            flag4 = np.setdiff1d(np.arange(1, 100), U2)

            if len(flag4) > 0:
                t2 = np.intersect1d(positions2[2, :], np.union1d(positions2[0, :], positions2[1, :]))
                id2 = np.random.choice(t2, len(flag4), replace=False)
                index4 = np.union1d(np.setdiff1d(positions2[2, :], t2[id2]), flag4)
                n_pop[ind[1], 200:300] = 0
                n_pop[ind[1], 200 + index4] = 1

    mutation_probability = 0.1  # 变异概率
    for i in range(num):
        if np.random.rand() < mutation_probability:
            pos1 = np.random.randint(1, dim // 3 + 1)
            pos2 = np.random.randint(1, dim // 3 + 1)
            t = np.random.randint(1, 4)

            while n_pop[i, pos1 + 100 * (t - 1)] == n_pop[i, pos2 + 100 * (t - 1)] or pos1 == pos2:
                pos2 = np.random.randint(1, dim // 3 + 1)

            # 将对应基因位置的二进制数反转
            if n_pop[i, pos1 + 100 * (t - 1)] == 0:
                n_pop[i, pos1 + 100 * (t - 1)] = 1
                n_pop[i, pos2 + 100 * (t - 1)] = 0
            else:
                n_pop[i, pos1 + 100 * (t - 1)] = 0
                n_pop[i, pos2 + 100 * (t - 1)] = 1

            # 小优化
            positions1 = np.zeros((3, 70))
            ran1 = n_pop[i, :]
            positions1[0, :] = np.where(ran1[0:100] == 1)[0]
            positions1[1, :] = np.where(ran1[100:200] == 1)[0]
            positions1[2, :] = np.where(ran1[200:300] == 1)[0][:70]
            U1 = np.union1d(positions1[0, :], np.union1d(positions1[1, :], positions1[2, :]))
            flag3 = np.setdiff1d(np.arange(1, 100), U1)

            if len(flag3) > 0:
                t2 = np.intersect1d(positions1[3, :], np.union1d(positions1[1, :], positions1[2, :]))
                id1 = np.random.choice(t2, len(flag3), replace=False)
                index3 = np.union1d(np.setdiff1d(positions1[3, :], t2[id1]), flag3)
                n_pop[i, 200:300] = 0
                n_pop[i, 200 + index3] = 1
    valhist_all[gen] = 100000 - np.max(fit)
    hist[gen] = np.min(valhist_all)
    print(hist[gen])
