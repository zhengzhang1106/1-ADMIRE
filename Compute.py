import Database


def compute_wavelength(links_physical):  # 计算使用波长
    wave, row, col = links_physical.shape
    wave_used_all = 0        # 记波长使用情况，一段时间路由策略不变，波长使用情况相同
    max_wave_num = int(wave/Database.time)
    for i in range(row):
        for j in range(col):
            if Database.graph_connect[i][j] == 1:
                for k in range(max_wave_num):
                    wave_used = 0
                    for t in range(Database.time):
                        if 0 <= links_physical[t * max_wave_num + k][i][j] < Database.wavelength_capacity:
                            wave_used = 1
                    wave_used_all += wave_used

    return wave_used_all


# 计算业务经过的跳数，一条源到目的的新建光路是一跳，只统计虚拟拓扑上的跳数
# 经过的跳数就是接入层节点数量除以2，接入层节点都是5的倍数
def compute_hop(route):
    virtual_hop_num = 0
    for i in range(len(route)):
        cur_node = route[i]
        if cur_node % 5 == 0:
            virtual_hop_num += 1
    virtual_hop_num /= 2
    # print("光路数量：",virtual_hop_num)
    return virtual_hop_num


if __name__ == '__main__':
    Database.links_physical[5][0][1] = 5
    print(compute_wavelength(Database.links_physical))
