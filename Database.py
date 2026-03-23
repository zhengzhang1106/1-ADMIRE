import numpy as np

# 网络和业务信息
job_number = 200  # 业务数量
wavelength_number = 3  # 波长数量
wavelength_capacity = 10  # 单波长容量
node_number = 9  # 节点个数
link_number = 12  # 网络中总的链路数
wavelength_number_all = link_number * wavelength_number * 2  # 72

time = 24  # 所有时刻数量，目前认为是24小时

# 0823更新，除了WBE其他边都有权重，为什么给Mux和DeMux添加权重之后，使用的波长就减少了
# 还有就是不同波长链路边权重不同的影响
# 辅助图权重
# 最小化波长链路数
GrmE_weight = 0
LPE_weight = 1
TxE_weight = 20
RxE_weight = 20
WLE_weight = 1000
MuxE_weight = 0
DeMuxE_weight = 0
WBE_weight = 0


# 最小化新建光路数
# GrmE_weight = 0
# LPE_weight = 1
# TxE_weight = 20
# RxE_weight = 20
# WLE_weight = 1000
# MuxE_weight = 0
# DeMuxE_weight = 0
# WBE_weight = 0


# 让DRL的动作空间在[-1,1]范围内
weight_max = 1  # 强化学习辅助图权重上限
weight_min = -1 # 强化学习辅助图权重上限
# increase = 5    # 权重增长步长

# 物理拓扑连接关系
graph_connect = np.array([[0, 1, 1, 0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=int)

# 物理链路及波长，考虑24小时，所以是72*9*9
links_physical = np.zeros((time*wavelength_number, node_number, node_number))


# wave = links_physical.shape[0]  # 维度
# row = links_physical.shape[1]  # 行
# col = links_physical.shape[2]  # 列
wave, row, col = links_physical.shape
# print('wave',wave)
for i in range(row):
    for j in range(col):
        if graph_connect[i][j] == 1:
            for k in range(wave):
                links_physical[k][i][j] = wavelength_capacity
        else:
            for k in range(wave):
                links_physical[k][i][j] = -1

# print(links_physical)


def clear(links):
    for i in range(row):
        for j in range(col):
            if graph_connect[i][j] == 1:
                for k in range(wave):
                    links[k][i][j] = wavelength_capacity
            else:
                for k in range(wave):
                    links[k][i][j] = -1.
    # print(links)
    return links


if __name__ == '__main__':
    clear(links_physical)
    print(wavelength_number_all)
    print(links_physical)
