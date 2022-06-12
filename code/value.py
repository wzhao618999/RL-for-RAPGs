import random
import math
from graph import Mdp
from utils import *

S = list()  # 非终止状态（需要计算价值的状态）
S2 = list()  # 所有状态（包括终止状态）
A = list()  # 所有动作
Ac = {}  # 动作（每个状态下可供选择的动作）

R = {}  # s,a,s_对应的奖励r
Pi = {}  # 策略（每个状态下选择相应的动作的概率）
P = {}  # 状态转移概率（每个状态下，执行相应的动作后，可能到达的后继状态对应的概率）

gamma = 1.0  # 折扣因子

# Value_Record1 = list()  # 记录各个状态的价值(法一)
# Value_Record2 = list()  # 记录各个状态的价值(法二)
Value_dict1 = dict()
Value_dict2 = dict()

Result_dict1 = dict()
Result_dict2 = dict()


# Threshold1 = 0.001  # 阈值上限
# Threshold2 = 0.0001  # 阈值下限

# Convergence1 = list()  # 记录每个状态的收敛速度，最后取最大值作为整体收敛速度
# Convergence2 = list()


def initialize(list, document):
    with open(document) as file:
        for line in file:
            line = line.split()
            print("line:", line)
            # 设置策略字典Pi
            set_pi(Pi, line[0], line[1], 0.5)
            # 设置状态转移字典P
            set_prob(P, line[0], line[1], line[2], float(line[3]))
            # 设置奖励字典R
            set_reward(R, line[0], line[1], line[2], float(line[4]))
            # 设置列表S
            if (line[0] not in S):
                S.append(line[0])
            S.sort()
            # 设置列表S2
            if (line[0] not in S2):
                S2.append(line[0])
            if (line[2] not in S2):
                S2.append(line[2])
            S2.sort()
            # 设置列表A
            if (line[1] not in A):
                A.append(line[1])
        # 打印
        print()
        print("非终止状态：", S)
        # 临时修改S
        for i in range(len(list)):
            if (list[i] in S): S.remove(list[i])
        print("修改后的非终止状态: ", S)
        print("所有状态：", S2)
        print("所有动作：", A)
        print()
        print("策略字典：")
        display_dict(Pi)
        print("状态转移概率字典：")
        display_dict(P)
        print("奖励字典：")
        display_dict(R)


# 设置动作字典
def set_action_dict(document):
    for i in range(len(S)):
        Ac.setdefault(S[i])

    # print(Ac.items())

    for i in Ac.keys():
        Ac[i] = []

    # print(Ac.items())

    with open(document) as file:
        for line in file:
            line = line.split()
            if (line[0] in S):
                if (line[1] not in Ac[line[0]]):
                    Ac[line[0]].append(line[1])

    print("动作字典：", Ac.items())
    print()


# 接收状态和动作，返回状态转移可到达的的后继状态的列表
def set_state_dict(s, a):
    l = list()
    for i in S2:
        if (get_prob(P, s, a, i) != 0):
            l.append(i)
    return l


# def calculate_method1():
#     # 先初始化一下Value_Record1和Convergence1
#     for i1 in range(len(S)):
#         Value_Record1.append(0)
#         Convergence1.append(0)
#     # 现在开始计算每一个状态的价值
#     for i2 in range(len(S)):
#         s = S[i2]
#         print("状态%s:" % s)
#         # 设置两个total_reward，一个考虑正负，一个不考虑正负，最后通过比较两个值的大小是否相等，可以判断是否全部到达B或全部不到达B
#         total_reward1 = 0
#         total_reward2 = 0
#         # 记录终止状态是集合B中的轮数
#         total_reward_b = 0
#         number_b = 0
#         # 记录终止状态不是集合B中的轮数
#         total_reward_not_b = 0
#         number_not_b = 0
#         # 记录每个状态的总迭代轮数
#         e_num = 0
#         # 循环迭代
#         while (True):
#             # 每一轮的总奖励
#             reward_for_each_episode = 0
#             # 一轮迭代
#             while (s in S):
#                 # 状态s下的动作选择
#                 if (len(Ac[s]) == 1):
#                     a = Ac[s][0]
#                 else:
#                     a1 = Ac[s][0]
#                     a2 = Ac[s][1]
#                     # 设置p1为（0,1）之间的随机数
#                     p1 = random.random()
#                     # 均一随机策略选择动作
#                     if (p1 >= get_pi(Pi, s, a1)):
#                         a = a1
#                     else:
#                         a = a2
#                 # 执行动作a后的状态转移
#                 l = set_state_dict(s, a)
#                 if (len(l) == 1):
#                     s_ = l[0]
#                 if (len(l) == 2):
#                     s_1 = l[0]
#                     s_2 = l[1]
#                     p2 = random.random()
#                     if (p2 <= get_prob(P, s, a, s_1)):
#                         s_ = s_1
#                     else:
#                         s_ = s_2
#                 reward = get_reward(R, s, a, s_)
#                 reward_for_each_episode += reward
#                 s = s_
#             # 每一轮迭代结束后判断
#             if (s != 's7'):
#                 total_reward1 += (reward_for_each_episode * (-1))
#                 total_reward_not_b += (reward_for_each_episode * (-1))
#                 number_not_b += 1
#             else:
#                 total_reward1 += reward_for_each_episode
#                 total_reward_b += reward_for_each_episode
#                 number_b += 1
#             total_reward2 += reward_for_each_episode
#             e_num += 1
#             if ((abs(total_reward1 / e_num - Value_Record1[i2]) < Threshold1)):
#                 Convergence1[i2] = e_num - 1
#                 break
#             Value_Record1[i2] = total_reward1 / e_num
#             s = S[i2]
#
#         if (total_reward1 == total_reward2):
#             print("全部到达B，奖励给系统，奖励值为：", total_reward1 / e_num)
#         elif ((total_reward1 + total_reward2) == 0):
#             print("全部不到达B，奖励给环境，奖励值为：", total_reward1 / e_num)
#         else:
#             if (total_reward1 >= 0):
#                 print("奖励给系统：奖励值为：", total_reward1 / e_num)
#             else:
#                 print("奖励给环境，奖励值为：", total_reward1 / e_num)
#
#         print("")
#
#     convergence1 = 0
#     for i3 in range(len(Convergence1)):
#         if (convergence1 < Convergence1[i3]):
#             convergence1 = Convergence1[i3]
#     print("状态价值列表：", Value_Record1)
#     print()
#     print("来看看每个状态迭代了几次：", Convergence1)
#     print()
#     print("法一经过%d轮收敛\n" % convergence1)


# def calculate_method2():
#     # 先初始化一下Value_Record2Convergence2
#     for i1 in range(len(S)):
#         Value_Record2.append(0)
#         Convergence2.append(0)
#     # 现在开始计算每一个状态的价值
#     for i2 in range(len(S)):
#         s = S[i2]
#         print("状态%s:" % s)
#         # 每一个状态的最终价值
#         # 设置两个total_reward，一个考虑正负，一个不考虑正负，最后通过比较两个值的大小是否相等，可以判断是否全部到达B或全部不到达B
#         total_reward1 = 0
#         total_reward2 = 0
#         # 记录终止状态是集合B中的episode
#         total_reward_b = 0
#         number_b = 0
#         # 记录终止状态不是集合B中的episode
#         total_reward_not_b = 0
#         number_not_b = 0
#         # 记录迭代轮数
#         e_num = 0
#         while (True):
#             # 每一轮的总奖励
#             reward_for_each_episode = 0
#             while (s in S):
#                 # 状态s下的动作选择
#                 if (len(Ac[s]) == 1):
#                     a = Ac[s][0]
#                 else:
#                     a1 = Ac[s][0]
#                     a2 = Ac[s][1]
#                     # 设置p1为（0,1）之间的随机数
#                     p1 = random.random()
#                     # 均一随机策略选择动作
#                     if (p1 >= get_pi(Pi, s, a1)):
#                         a = a1
#                     else:
#                         a = a2
#                 # 执行动作a后的状态转移
#                 l = set_state_dict(s, a)
#                 if (len(l) == 1):
#                     s_ = l[0]
#                 if (len(l) == 2):
#                     s_1 = l[0]
#                     s_2 = l[1]
#                     p2 = random.random()
#                     if (p2 <= get_prob(P, s, a, s_1)):
#                         s_ = s_1
#                     else:
#                         s_ = s_2
#                 reward = get_reward(R, s, a, s_)
#                 reward_for_each_episode += reward
#                 s = s_
#             if (s != 's7'):
#                 total_reward1 += (reward_for_each_episode * (-1))
#                 total_reward_not_b += (reward_for_each_episode * (-1))
#                 number_not_b += 1
#             else:
#                 total_reward1 += reward_for_each_episode
#                 total_reward_b += reward_for_each_episode
#                 number_b += 1
#             total_reward2 += reward_for_each_episode
#             e_num += 1
#             if (number_b == 0):
#                 help_num = total_reward_not_b / number_not_b
#             elif (number_not_b == 0):
#                 help_num = total_reward_b / number_b
#             else:
#                 help_num = total_reward_b / number_b + total_reward_not_b / number_not_b
#             if ((abs(help_num - Value_Record2[i2]) < Threshold1)
#                     and (abs(help_num - Value_Record2[i2]) > Threshold2)):
#                 Convergence2[i2] = e_num - 1
#                 break
#             Value_Record2[i2] = help_num
#             # 这里一定要重置s
#             s = S[i2]
#
#         if (total_reward1 == total_reward2):
#             print("全部到达B，奖励给系统，奖励值为：", help_num)
#         elif ((total_reward1 + total_reward2) == 0):
#             print("全部不到达B，奖励给环境，奖励值为：", help_num)
#         else:
#             if (total_reward1 >= 0):
#                 print("奖励给系统：奖励值为：", help_num)
#             else:
#                 print("奖励给环境：奖励值为：", help_num)
#
#         print("")
#
#     convergence2 = 0
#     for i in range(len(Convergence2)):
#         if (convergence2 < Convergence2[i]):
#             convergence2 = Convergence2[i]
#     print("状态价值列表：", Value_Record2)
#     print()
#     print("来看看每个状态迭代了几次：", Convergence2)
#     print()
#     print("法二经过%d轮收敛\n" % convergence2)


def calculate(episode, list, index):
    Value_Record1 = []
    Value_Record2 = []
    for i in range(len(S)):
        s = S[i]
        if (index == 1):
            print("状态%s:" % s)
        # 每一个状态的最终价值
        # 设置两个total_reward，一个考虑正负，一个不考虑正负，最后通过比较两个值的大小是否相等，可以判断是否全部到达B或全部不到达B
        total_reward1 = 0
        total_reward2 = 0
        # 记录终止状态是集合B中的episode
        total_reward_b = 0
        number_b = 0
        # 记录终止状态不是集合B中的episode
        total_reward_not_b = 0
        number_not_b = 0
        for j in range(episode):
            # 每一轮的总奖励
            reward_for_each_episode = 0
            while (s in S):
                # 状态s下的动作选择
                if (len(Ac[s]) == 1):
                    a = Ac[s][0]
                else:
                    a1 = Ac[s][0]
                    a2 = Ac[s][1]
                    # 设置p1为（0,1）之间的随机数
                    p1 = random.random()
                    # 均一随机策略选择动作
                    if (p1 >= get_pi(Pi, s, a1)):
                        a = a1
                    else:
                        a = a2
                # 执行动作a后的状态转移
                l = set_state_dict(s, a)
                if (len(l) == 1):
                    s_ = l[0]
                if (len(l) == 2):
                    s_1 = l[0]
                    s_2 = l[1]
                    p2 = random.random()
                    if (p2 <= get_prob(P, s, a, s_1)):
                        s_ = s_1
                    else:
                        s_ = s_2
                reward = get_reward(R, s, a, s_)
                reward_for_each_episode += reward
                s = s_
            if (s not in list):
                total_reward1 += (reward_for_each_episode * (-1))
                total_reward_not_b += (reward_for_each_episode * (-1))
                number_not_b += 1
            else:
                total_reward1 += reward_for_each_episode
                total_reward_b += reward_for_each_episode
                number_b += 1
            total_reward2 += reward_for_each_episode
            # 这里一定要重置s
            s = S[i]

        if (index == 1):
            if (total_reward1 == total_reward2):
                print("全部到达B，奖励给系统，奖励值为：", total_reward1 / episode)
                Value_Record1.append(total_reward1 / episode)
            elif ((total_reward1 + total_reward2) == 0):
                print("全部不到达B，奖励给环境，奖励值为：", total_reward1 / episode)
                Value_Record1.append(total_reward1 / episode)
            else:
                if (total_reward1 >= 0):
                    print("奖励给系统：奖励值为：", total_reward1 / episode)
                    Value_Record1.append(total_reward1 / episode)
                else:
                    print("奖励给环境：奖励值为：", total_reward1 / episode)
                    Value_Record1.append(total_reward1 / episode)

            print("============================")

            if (total_reward1 == total_reward2):
                print("全部到达B，奖励给系统，奖励值为：", total_reward1 / episode)
                Value_Record2.append(total_reward1 / episode)
            elif ((total_reward1 + total_reward2) == 0):
                print("全部不到达B，奖励给环境，奖励值为：", total_reward1 / episode)
                Value_Record2.append(total_reward1 / episode)
            else:
                if (total_reward1 >= 0):
                    print("奖励给系统：奖励值为：", total_reward_b / number_b + total_reward_not_b / number_not_b)
                    Value_Record2.append(total_reward_b / number_b + total_reward_not_b / number_not_b)
                else:
                    print("奖励给环境，奖励值为：", total_reward_b / number_b + total_reward_not_b / number_not_b)
                    Value_Record2.append(total_reward_b / number_b + total_reward_not_b / number_not_b)

            print("")

        else:
            if (total_reward1 == total_reward2):
                Value_Record1.append(total_reward1 / episode)
            elif ((total_reward1 + total_reward2) == 0):
                Value_Record1.append(total_reward1 / episode)
            else:
                if (total_reward1 >= 0):
                    Value_Record1.append(total_reward1 / episode)
                else:
                    Value_Record1.append(total_reward1 / episode)

            if (total_reward1 == total_reward2):
                Value_Record2.append(total_reward1 / episode)
            elif ((total_reward1 + total_reward2) == 0):
                Value_Record2.append(total_reward1 / episode)
            else:
                if (total_reward1 >= 0):
                    Value_Record2.append(total_reward_b / number_b + total_reward_not_b / number_not_b)
                else:
                    Value_Record2.append(total_reward_b / number_b + total_reward_not_b / number_not_b)
    return Value_Record1, Value_Record2


def result(episode, list, document):
    VR1, VR2 = calculate(episode, list, 1)
    # 列表转字典，方便对应状态和价值
    for i in range(len(S2)):
        Value_dict1[S2[i]] = 0
        Value_dict2[S2[i]] = 0
    for i in range(len(S)):
        Value_dict1[S[i]] = VR1[i]
        Value_dict2[S[i]] = VR2[i]
    print("法一求价值：")
    print("Value_dict1.items():", Value_dict1.items())
    print("法二求价值：")
    print("Value_dict2.items():", Value_dict2.items())
    print()

    # 求概率字典
    m = Mdp()
    m.ReadMdpFromFile(document)
    m.dot(set())
    B = m.getB(list)
    m.F_pr(B)
    print("概率字典：", m.pr.items())

    # 加权求最后结果
    for k, v in Value_dict1.items():
        Result_dict1[k] = v * m.pr[k]
        Result_dict2[k] = Value_dict2[k] * m.pr[k]
    print()
    print("最后结果：")
    print("Result_dict1.items():", Result_dict1.items())
    print("Result_dict2.items():", Result_dict2.items())


def compare(list):
    # 法一情况
    fluctuation = fluctuation_calculate1(500, 5, list)
    print("法一500次迭代波动情况，均方误差：", fluctuation)
    fluctuation = fluctuation_calculate1(1000, 5, list)
    print("法一1000次迭代波动情况，均方误差：", fluctuation)
    fluctuation = fluctuation_calculate1(5000, 5, list)
    print("法一5000次迭代波动情况，均方误差：", fluctuation)
    fluctuation = fluctuation_calculate1(10000, 5, list)
    print("法一10000次迭代波动情况，均方误差：", fluctuation)
    convergence = convergence_calculate1(500, 5, list)
    print("法一500次迭代收敛情况，均方误差：", convergence)
    convergence = convergence_calculate1(1000, 5, list)
    print("法一1000次迭代收敛情况，均方误差：", convergence)
    convergence = convergence_calculate1(5000, 5, list)
    print("法一5000次迭代收敛情况，均方误差：", convergence)
    convergence = convergence_calculate1(10000, 5, list)
    print("法一10000次迭代收敛情况，均方误差：", convergence)

    # 法二情况
    fluctuation = fluctuation_calculate2(500, 5, list)
    print("法二500次迭代波动情况，均方误差：", fluctuation)
    fluctuation = fluctuation_calculate2(1000, 5, list)
    print("法二1000次迭代波动情况，均方误差：", fluctuation)
    fluctuation = fluctuation_calculate2(5000, 5, list)
    print("法二5000次迭代波动情况，均方误差：", fluctuation)
    fluctuation = fluctuation_calculate2(10000, 5, list)
    print("法二10000次迭代波动情况，均方误差：", fluctuation)
    convergence = convergence_calculate2(500, 5, list)
    print("法二500次迭代收敛情况，均方误差：", convergence)
    convergence = convergence_calculate2(1000, 5, list)
    print("法二1000次迭代收敛情况，均方误差：", convergence)
    convergence = convergence_calculate2(5000, 5, list)
    print("法二5000次迭代收敛情况，均方误差：", convergence)
    convergence = convergence_calculate2(10000, 5, list)
    print("法二10000次迭代收敛情况，均方误差：", convergence)


def fluctuation_calculate1(episode, k, list):
    fluctuation = 0
    for i in range(k):
        method1_list11, _ = calculate(episode, list, 0)
        method1_list12, _ = calculate(episode, list, 0)
        # print(method1_list11)
        # print(method1_list12)
        fluctuation += mean_square_error(method1_list11, method1_list12)
    fluctuation /= k
    return fluctuation


def fluctuation_calculate2(episode, k, list):
    fluctuation = 0
    for i in range(k):
        _, method2_list11 = calculate(episode, list, 0)
        _, method2_list12 = calculate(episode, list, 0)
        # print(method1_list11)
        # print(method1_list12)
        fluctuation += mean_square_error(method2_list11, method2_list12)
    fluctuation /= k
    return fluctuation


def convergence_calculate1(episode, k, list):
    convergence = 0
    for i in range(k):
        method1_list11, _ = calculate(episode, list, 0)
        method1_list12, _ = calculate(episode + 1, list, 0)
        # print(method1_list11)
        # print(method1_list12)
        convergence += mean_square_error(method1_list11, method1_list12)
    convergence /= k
    return convergence


def convergence_calculate2(episode, k, list):
    convergence = 0
    for i in range(k):
        _, method2_list11 = calculate(episode, list, 0)
        _, method2_list12 = calculate(episode + 1, list, 0)
        # print(method1_list11)
        # print(method1_list12)
        convergence += mean_square_error(method2_list11, method2_list12)
    convergence /= k
    return convergence


def mean_square_error(list1, list2):
    result = 0
    for i in range(len(list1)):
        result += math.pow((list1[i] - list2[i]), 2)
    return result


if __name__ == "__main__":
    initialize(["s6", "s7", "s8", "s9", "s10", "s11", "s13"], "mdp2")
    set_action_dict("mdp2")
    result(10000, ["s6", "s7", "s8", "s9", "s10", "s11", "s13"], "mdp2")
    print()
    compare(["s6", "s7", "s8", "s9", "s10", "s11", "s13"])
    # calculate_method1()
    # calculate_method2()
