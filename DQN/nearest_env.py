import functools
import math
from operator import itemgetter

import numpy as np
import pandas as pd
import random
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from time import time
import  datetime


def env():
    '''
    env函数通常在默认情况下将环境包裹在包装器中。
    你可以在开发者文档中找到这些方法的完整文档
    的完整文档，在开发者文档的其他地方。
    '''
    env = raw_env()
    # 这个包装器只适用于向终端打印结果的环境
    env = wrappers.CaptureStdoutWrapper(env)
    # 这个包装器有助于离散行动空间的错误处理
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # 提供广泛的有用的用户错误。
    # 强烈推荐
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env():
    '''
    为了支持AEC API，raw_env()函数只是使用from_parallel
    函数来将ParallelEnv转换为AEC环境
    '''
    env = parallel_env()
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, info_callback=None, observation_callback=None):

        # 时间切片
        self.t_s = 0.5
        # 最大SOC
        self.max_soc = 60
        # 单位时间请求数
        self.n_r = 6
        # 充电速率 charging_speed 20 kw/h
        self.c_s = 20
        # 经纬度
        self.w = 100
        self.h = 100
        # 速度
        self.speed = 20
        # 时间步骤
        self.STEP = 0
        # 等待?
        self.waiting = False

        # 位置信息
        # 汽车位置
        self.l_q = []
        # 汽车SOC
        self.s_q = []
        # 预计抵达时间
        self.a_t = []

        # 充电站数量
        self.n_lc = 10
        # 每个充电站充电桩数量
        self.n_c = 10
        # 充电桩位置
        self.l_c = []
        # 充电站使用情况 like [[1,0,0,0,1,1], []]
        self.s_c = []
        # 具体情况
        self.inf_c = []
        # 各充电站空闲充电桩数量
        self.r_n_c = []
        # 充电站等待队列 charging_waiting_list
        self.c_w_l = []

        # 各充电站 负载历史
        self.his_load = np.zeros(self.n_lc)
        self.waiting_time_his = 0
        self.Loads = []

        self.update_rest = []

        self.check_log = False

# -----------------------------------------------------
        self.possible_agents = ["player_" + str(r) for r in range(self.n_r)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.info_callback = info_callback
        self.observation_callback = observation_callback
        # self.action_space = []
        # self.observation_space = []
        # for i in range(self.n_r):
        #     total_action_space = []
        #     action_space = Discrete(self.n_lc)
        #     total_action_space.append(action_space)
        #     self.action_space.append(action_space)
        #
        #     # obs 负载 + 空闲充电桩
        #     observation_space = Box(low=np.array([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]), high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]), dtype=np.float64)
        #     self.observation_space.append(observation_space)
            # self.observation_space = Discrete(10)
        self.action_space = Discrete(self.n_lc)
        self.observation_space = Box(low=np.array([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]), high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]), dtype=np.float64)
        # self.observation_space = Box(low=np.array([-100, -100]), high=np.array([100, 100]), dtype=np.float64)

    def render(self, mode="human"):
        if self.check_log:
            print('rest:', self.r_n_c)
            print('waiting list:', self.c_w_l)
            print('预计抵达时间 at', self.a_t)
        else:
            pass

    def close(self):
        pass

    def reset(self):
        # 1.数组清空
        self.inf_c = []             # [[[1, 22, time], [0, 10, time]], [[1, 22, time], [0, 10, time], [0, 10, time]]]
        self.s_c = []
        self.r_n_c = []             # [1,2,3,-2,3]
        self.l_c = []
        self.l_q = []
        self.s_q = []
        self.a_t = []
        self.c_w_l = []
        self.STEP = 0
        self.Load = 0
        self.waiting = False

        self.n_r = np.random.randint(150, 160)

        # 2.充电站信息初始化

        for i in range(self.n_lc):
            info_c = []
            info_s = []
            # 充电情况
            charging_list = np.random.randint(0, 2, self.n_c)
            self.s_c.append(charging_list)
            # print(self.s_c)

            # 具体充电情况
            # 开始时间
            start_time = datetime.datetime.now()
            info = self.s_c[i]
            for piles in info:
                if piles:
                    start_soc = np.random.randint(0,  self.max_soc)
                    charging_time = (self.max_soc - start_soc) / 20
                else:
                    start_soc = 0
                    charging_time = 0
                end_time = start_time + datetime.timedelta(hours=charging_time)
                info_c.append([piles, start_soc, start_time, end_time])
                # info_s.append(info_c)
            self.inf_c.append(info_c)

            # 空闲充电桩数量
            free_charging = np.where(charging_list, 0, 1)
            self.r_n_c.append(sum(free_charging))
            # 等待队列
            self.c_w_l.append([0, []])
            # 位置
            self.l_c.append([np.random.randint(0, self.w), np.random.randint(0, self.h)])

        # 3.电车信息初始化
        for i in range(self.n_r):
            # SOC reset
            self.s_q.append(np.random.randint(0, self.max_soc))

            # 位置
            self.l_q.append([np.random.randint(0, self.w), np.random.randint(0, self.h)])

            # 抵达时间
            arrival_time_list = []
            for j in range(len(self.l_c)):
                dis = math.sqrt((self.l_q[-1][0] - self.l_c[j][0])**2 + (self.l_c[-1][1] - self.l_c[j][1])**2)
                arrival_time = dis / self.speed
                arrival_time_list.append(arrival_time)
            self.a_t.append(arrival_time_list)
        # print(len(self.a_t))


        # rest_num = 0
        # for i in range(len(self.r_n_c)):
        #     rest_num += self.r_n_c[i]
        load = 0
        for rest in self.r_n_c:
            # 负载奖励
            load -= 1 / rest
        # obs[0] = self.a_t[1][1]
        # obs[1] = load
        # obs = 空闲和 + 负载和
        # -----!-----
        # obs = [0, 0]
        # obs[0] = self.l_q[0][0]
        # obs[1] = self.l_q[0][1]
        # -----!-----
        obs = self.r_n_c
        # obs = 0
        # obs[2] = self.l_c[0][0]
        # obs[3] = self.l_c[0][1]
        self.Loads.append(load)
        # obs[2] = load

        if self.check_log:
            print('----rest over----')
        # print('info total', self.inf_c)
        # print('----rest over----')
        return obs

    def step(self, action):
        # 1.初始化序列
        # 2.检查序列
        rest, _ = self.check_charging(action)
        # print(self.c_w_l)

        # self.r_n_c[action] -= 1
        # 3.奖励
        arrival_time_list = self.a_t[self.STEP]
        rewards, load = self.rewards(action, arrival_time_list, rest)
        # self.Loads = load

        # print(action, self.r_n_c)
        # self.r_n_c 空闲序列 对应 -1; self.c_w_l 是否等待：等待序列 +1;


        # 5.obs = 空闲和 + 负载和
        if self.STEP == 0:
            self.Loads.append(load)
        else:
            self.Loads[0] = self.Loads[1]
            self.Loads[1] = load
        # rest_num = 0
        # for i in range(len(self.r_n_c)):
        #     rest_num += self.r_n_c[i]
        # obs = [0, 0]
        # obs[0] = arrival_time_list[action]
        # obs[1] = load

        # obs = [0, 0]
        # obs[0] = self.l_q[self.STEP+1][0]
        # obs[1] = self.l_q[self.STEP+1][1]
        # obs[2] = self.l_c[action][0]
        # obs[3] = self.l_c[action][1]
        # obs[4] = load

        # 6.done
        self.STEP += 1
        if self.STEP == len(self.s_q):
            done = True
        else:
            done = False

        # -----obs------
        # obs = [0, 0]
        # if done:
        #     obs[0] = self.l_q[self.STEP-1][0]
        #     obs[1] = self.l_q[self.STEP-1][1]
        # else:
        #     obs[0] = self.l_q[self.STEP][0]
        #     obs[1] = self.l_q[self.STEP][1]
        obs = self.r_n_c
        # obs = action
        # obs[2] = load
        # 7.his

        return obs, rewards, done, {}

    def rewards(self, action, arrival_time, weather_rest):
        # 1.抵达时间奖励
        min_time = arrival_time[0]
        for i in range(len(arrival_time)):
            if arrival_time[i] < min_time:
                min_time = arrival_time[i]
        if min_time == 0:
            if arrival_time[action] == min_time:
                r_arrival_time = -1
            else:
                r_arrival_time = -arrival_time[action] / 10
            # print(arrival_time)
        else:
            r_arrival_time = -arrival_time[action] / min_time

        # 2.等待时间奖励+负载奖励
        r_load = 0
        r_waiting = 0
        # 空闲 负载奖励 = sum(1/rest[action]); 等待奖励 = 0
        # print(weather_rest)
        if weather_rest:
            for rest in self.r_n_c:
                # 负载奖励
                # print(rest)
                if rest == 0:
                    r_load -= 0
                else:
                    r_load -= 1 / rest
            r_waiting = 0
            self.waiting_time_his = 0
        # 等待
        else:
            for wait in self.c_w_l:
                if wait[0] != 0:
                    r_load -= 1 / wait[0]
                else:
                    r_load = r_load
        # self.r_n_c[action] = 0

            # 等待时间奖励 等待队列中的位置-->  对应到详细信息中的结束时间排序
            n_wait = len(self.c_w_l[action][1])
            n_wait -= 1
            soc1 = self.c_w_l[action][1][n_wait]
            # 如果等待的数量超过10，
            # print('n_wait1:', n_wait)
            if n_wait > 9:
                soc = 0
                while n_wait > 9:
                    soc += (self.max_soc - self.c_w_l[action][1][n_wait])
                    n_wait -= 10
                # print('n_wait2:', n_wait)
                charging_time = (soc - soc1) / (20*60*60)
                wait_info = sorted(self.inf_c[action], key=itemgetter(3))
                waiting_time = wait_info[n_wait][3].timestamp() - datetime.datetime.now().timestamp() + charging_time
                # print(waiting_time)
                # waiting_time = wait_info[n_wait][3].timestamp() - datetime.datetime.now().timestamp()

            else:
                wait_info = sorted(self.inf_c[action], key=itemgetter(3))
                # print(n_wait, wait_info)
                # 前一个的结束时间为下一个的开始时间
                waiting_time = wait_info[n_wait][3].timestamp() - datetime.datetime.now().timestamp()
            # print(abs(waiting_time))
            r_waiting = - abs(waiting_time)
            self.waiting_time_his = np.float(abs(waiting_time))
            # r_waiting 0.04590892791748047
        # print('r_arrival_time', r_arrival_time)
        # print('r_waiting', r_waiting / 2000)
        # print('r_load', r_load)
        R = r_arrival_time + r_waiting/2000
        return R, r_load

    def history(self):
        # print(self.waiting_time_his)
        return self.Loads[1], self.waiting_time_his

    # 检查全局充电桩详细信息并跟新
    def check_charging(self, action):
        rest_pile_id = 0
        stations_id = 0
        for stations in self.inf_c:
            # 检查空余充电桩&清空空闲数据
            piles_id = 0
            for piles in stations:
                # 1.判断是否空闲
                # print('station_id',  stations_id, 'piles_id', piles_id, 'piles_info:', piles)
                if piles[0] == 1:
                    # 正充电
                    # 2.检查充电状态
                    # print(piles[3] > datetime.datetime.now())
                    if (piles[3] != 0) & (piles[3] < datetime.datetime.now()):
                        # 已完成充电 清空info状态 [piles, start_soc, start_time, end_time]
                        piles[0] = 0
                        piles[1] = 0
                        piles[2] = 0
                        piles[3] = 0
                        # 是否存在等待序列
                        if self.c_w_l[stations_id][0] > 0:
                            # 等待序列-->充电序列
                            # 跟新充电序列 [piles, start_soc, start_time, end_time]
                            soc = self.c_w_l[stations_id][1][0]
                            charging_time = (self.max_soc - soc) / 20
                            end_time = datetime.datetime.now() + datetime.timedelta(microseconds=charging_time)
                            # end_time = a.name.strftime("%Y-%m-%d %H:%M:%S")
                            # end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                            # print(type(end_time))
                            piles[0] = 1
                            piles[1] = self.c_w_l[stations_id][1][0]
                            piles[2] = datetime.datetime.now()
                            piles[3] = end_time
                            # print(piles[2], piles[3])
                            # 删除等待序列信息
                            # 等待数量-1 & 删除等待的SOC
                            self.c_w_l[stations_id][0] -= 1
                            self.c_w_l[stations_id][1].pop(0)
                            # self.r_n_c[stations_id] += 1
                        else:
                            # 没有等待的
                            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            self.r_n_c[stations_id] += 1
                    # else:
                    #     pass
                else:
                    # 空闲
                    piles[0] = 0
                    piles[1] = 0
                    piles[2] = 0
                    piles[3] = 0
                piles_id += 1
            # 若是选择的站点
            if stations_id == action:
                rest = False
                # 空闲
                rest_pile_id = 0
                for piles in stations:
                    if piles[0] == 0:
                        rest = True
                        soc = self.s_q[self.STEP]
                        charging_time = (self.max_soc - soc) / 20
                        end_time = datetime.timedelta(microseconds=charging_time)
                        # 跟新详细信息列表
                        self.inf_c[action][rest_pile_id][0] = 1
                        self.inf_c[action][rest_pile_id][1] = soc
                        self.inf_c[action][rest_pile_id][2] = datetime.datetime.now()
                        self.inf_c[action][rest_pile_id][3] = datetime.datetime.now() + end_time
                        # self.r_n_c[action] -= 1
                        break
                    rest_pile_id += 1
                # 等待队列跟新
                if not rest:
                    self.c_w_l[stations_id][0] += 1
                    self.c_w_l[stations_id][1].append(self.s_q[self.STEP])
                    if self.r_n_c[action] > 0:
                        self.r_n_c[action] -= 1


            # 当前选择站点的id
            stations_id += 1
            # station_check:是否有空位<--->rest_pile_id:空位id
            # print(rest)
        return rest, rest_pile_id




    # 5.4 Ques
    # load 计算错误


# if __name__ == '__main__':
#     env = parallel_env()
#     obs = env.reset()
#
#     env.step(obs)