# coding=UTF-8
'''
coding=utf-8
@Descripttion: 给定博弈MDP, 求解GF B，F B
@version: 
@Author: Ares
@Date: 2019-08-18 14:17:13
@LastEditors: Ares
@LastEditTime: 2019-10-10 16:26:37
'''
import logging, random
import re
import sys
from copy import deepcopy, copy
from fractions import Fraction

from graphviz import Digraph
from z3 import *
from queue import Queue  # LILO队列

'''
@name: MAX, MIN
@msg: 在 z3 变量集中寻找最大/最小
@param {z3 变量集} 
@return: 最大/最小
'''


def MAX(lst):
    ret = lst[0]
    for i in lst:
        ret = If(ret > i, ret, i)
    return ret


def MIN(lst):
    ret = lst[0]
    for i in lst:
        ret = If(ret < i, ret, i)
    return ret


eps = 0.00000000000001

'''
@msg: MDP 节点有标签, 节点类型, 以及关于节点的迁移.{'act1': [(t1, p1),(t2,p2)...], 'act2':[...], ...}
'''


class MdpNode:
    def __init__(self, label, kind):
        self.actions = {}
        self.label = label
        self.kind = kind  # 属于哪种节点
        self.idx = -1  # 根据 label 排序后的下标
        self.canReachB = -1  # 标记是否能到达 B 集合

    def __str__(self):
        ret = ''
        for k in self.actions.keys():
            for out in self.actions[k]:
                ret += str(self.kind) + '\t' + self.label + '\t' + k + '\t' + out[0].label + '\t' + str(out[1]) + '\n'
        return ret

    def __repr__(self):
        return self.label

    def __eq__(self, other):
        return self.label == other.label and self.kind == other.kind

    def __hash__(self):
        return hash(str(self.kind) + ',' + str(self.label))


'''
@msg: 只有联通信息的图
'''


class graph:
    def __init__(self):
        self.nodes = {}

    def _addEdge(self, x, y):
        if x not in self.nodes.keys():
            self.nodes[x] = set()
        self.nodes[x].add(y)


'''
@msg: 不动点迭代
'''


class FixPoint:
    def __init__(self):
        self.pre = None

    def advance(self, cur):
        if cur == self.pre:
            return False
        else:
            self.pre = copy.copy(cur)
            return True


'''
@msg: 包含 MDP 节点
'''


class Mdp:
    def __init__(self):
        # 这三个都是set()类的实例
        self.MdpNodes = set()
        self.InitNodes = set()
        self.GoalNodes = set()
        self.reverse_undergraph = graph()
        self.value = dict()
        self.pr = dict()

    def _KindOfNode(self, label):
        num = int(re.findall(r'\d+', label)[0])
        if num % 2 == 0:
            return 'env'
        return 'sys'

    # labelX就是图中边的起始节点，Act是动作，labelY是末尾节点，Pr是状态转移概率，Reward是奖励
    def _addEdge(self, labelX, Act, labelY, Pr, Reward, degreeNotZero):
        NodeX = None
        NodeY = None
        # self.MdpNodes中一开始没有节点，节点是随着ReadMdpFromFile函数中那个for循环慢慢出现的
        for node in self.MdpNodes:
            if node.label == labelX:
                NodeX = node
            if node.label == labelY:
                NodeY = node
        if NodeX is None:
            # MdpNode是一个类，描述节点的类，在这里，NodeX是该类创建的实例
            NodeX = MdpNode(labelX, self._KindOfNode(labelX))
            self.MdpNodes.add(NodeX)  # 就是这里，为MdpNodes中添加节点
        if NodeY is None:
            NodeY = MdpNode(labelY, self._KindOfNode(labelY))
            self.MdpNodes.add(NodeY)
        if Act not in NodeX.actions.keys():
            NodeX.actions[Act] = []
        NodeX.actions[Act].append((NodeY, Pr, Reward))
        self.reverse_undergraph._addEdge(NodeY, NodeX)
        degreeNotZero.add(NodeX)

    def ReadMdpFromFile(self, filename):
        degreeNotZero = set()
        with open(filename) as file:
            for line in file:
                line = line.split()
                line[3] = float(Fraction(line[3]))
                # 最关键是调用了_addEdge函数，那么_addEdge函数中到底发生了什么？（注意这里是循环调用）
                self._addEdge(line[0], line[1], line[2], line[3], line[4], degreeNotZero)
        # 这个函数用于增加一些节点关系，使得图中没有出度为0的节点
        # 虽然其中的代码不需要仔细看，但是从中可以理解到self.MdpNodes、degreeNotZero和degreeIsZero中分别都是哪些节点
        self.CompleteMdp(degreeNotZero)

    def ReadMdpValueFromFile(self, filename):
        with open(filename) as file:
            for line in file:
                line = line.split()
                self.value[line[0]] = line[1]

    '''
    @msg: 将MDP补全，不再有没有后继的节点
    '''

    def CompleteMdp(self, degreeNotZero):

        env_Failed = set()
        sys_Failed = set()
        sysSuccNode = None
        sysFailNode = None
        envSuccNode = None
        envFailNode = None

        degreeIsZero = self.MdpNodes - degreeNotZero
        for node in degreeIsZero:

            # print("出度为0：", node.label)
            # print("node.label:", node.label)
            # print("node.actions:", node.actions)
            # print("type of the node.actions:", type(node.actions))  # node.actions是字典类型
            # print("node.actions.keys:", node.actions.keys)
            # print("node.kind", node.kind)  # 方的节点是系统类型(kind = sys)，圆的是环境类型(kind = env)

            if node.kind == 'sys':
                sys_Failed.add(node)
            else:
                env_Failed.add(node)

        if len(env_Failed) != 0:
            sysSuccNode = MdpNode('sysSucc', 'sys')
            envFailNode = MdpNode('envFail', 'env')
            sysSuccNode.actions['true'] = []
            envFailNode.actions['true'] = []
            sysSuccNode.actions['true'].append((envFailNode, 1.0))
            envFailNode.actions['true'].append((sysSuccNode, 1.0))
            self.reverse_undergraph._addEdge(envFailNode, sysSuccNode)
            self.reverse_undergraph._addEdge(sysSuccNode, envFailNode)

            for node in env_Failed:
                node.actions['true'] = []
                node.actions['true'].append((sysSuccNode, 1.0))
                self.reverse_undergraph._addEdge(sysSuccNode, node)
            self.MdpNodes.add(sysSuccNode)
            self.MdpNodes.add(envFailNode)
        if len(sys_Failed) != 0:
            sysFailNode = MdpNode('sysFail', 'sys')
            envSuccNode = MdpNode('envSucc', 'env')
            sysFailNode.actions['true'] = []
            envSuccNode.actions['true'] = []
            sysFailNode.actions['true'].append((envSuccNode, 1.0))
            envSuccNode.actions['true'].append((sysFailNode, 1.0))
            self.reverse_undergraph._addEdge(sysFailNode, envSuccNode)
            self.reverse_undergraph._addEdge(envSuccNode, sysFailNode)

            for node in sys_Failed:
                node.actions['true'] = []
                node.actions['true'].append((envSuccNode, 1.0))
                self.reverse_undergraph._addEdge(envSuccNode, node)
            self.MdpNodes.add(sysFailNode)
            self.MdpNodes.add(envSuccNode)

    def __str__(self):
        ret = ''
        for node in self.MdpNodes:
            ret += str(node)
        return ret

    '''
    @msg: 将mdp转为dot输出
    '''

    def dot(self, B):
        g = Digraph(comment='MDP')
        # print("self.value.items():", self.value.items())
        for node in self.MdpNodes:
            # print("node.label:", node.label)
            # print("self.value.get(node.label):", self.value.get(node.label))
            if node.kind == 'sys':
                g.node(name=node.label, label=node.label, shape="box")
            if node in B:
                g.node(name=node.label, label=node.label, color="green")

        for node in self.MdpNodes:
            for k in node.actions.keys():
                for out in node.actions[k]:
                    if len(out) == 2:
                        g.edge(node.label, out[0].label, k + '/' + str(out[1]))
                    if len(out) == 3:
                        g.edge(node.label, out[0].label, k + '/' + str(out[1]) + '/' + out[2])

        g.view()

    # 迭代法求解 FB
    def method_iter(self, Value, B):
        LastVal = {}
        for node in self.MdpNodes:
            LastVal[node] = Value[node]
        max_or_min = None
        cnt = 0
        while True:
            for node in self.MdpNodes:
                if node in B:
                    Value[node] = B[node]
                    continue
                if node.kind == 'env':
                    max_or_min = min
                else:
                    max_or_min = max
                ret = []
                for k in node.actions:
                    tmp = 0.0
                    for out in node.actions[k]:
                        tmp += LastVal[out[0]] * out[1]
                    ret.append(tmp)
                if len(ret) == 0:
                    Value[node] = LastVal[node]
                else:
                    Value[node] = max_or_min(ret)
            cnt += 1
            finish = True
            for node in self.MdpNodes:
                if abs(LastVal[node] - Value[node]) > eps:
                    finish = False
                    break
            if finish:
                return Value
            for node in self.MdpNodes:
                LastVal[node] = Value[node]

    # sat 求解 FB
    def method_sat(self, B, up, low):
        s = Solver()
        x = {}  # x[i]为s[i]对应的可达概率
        for node in self.MdpNodes:
            xi = Real(node.label)
            x[node] = xi
            s.add(xi >= 0.0)
            s.add(xi <= 1.0)
        for node in self.MdpNodes:
            if node in B:  # 在 B 集合中的概率都是 1.0
                s.add(x[node] == B[node])
                continue
            if node.kind == 'env':
                f = MIN
            else:
                f = MAX
            ch = []
            for k in node.actions:
                tmp = Real('tmp')
                tmp = 0.0
                for out in node.actions[k]:
                    tmp = tmp + x[out[0]] * out[1]
                ch.append(tmp)
            if len(ch) == 0: continue  # 没有出边
            s.add(x[node] == f(ch))

        # 找3个指派
        cnt = 3
        ret = []
        while s.check() == sat and cnt > 0:
            cnt -= 1
            m = s.model()
            # 验证是否在最小不动点和最大不动点之间
            flg_fail = 0
            for node in self.MdpNodes:
                vl = float(Fraction(str(m[x[node]])))
                if vl > up[node] + eps or vl + eps < low[node]:
                    flg_fail = 1
                    print('fail')
                    break
            if flg_fail: continue
            ret.append(m)
            nc = []
            for node in self.MdpNodes:
                nc.append(x[node] != m[x[node]])
            s.add(Or(nc))
        return ret, len(ret)

    def _advanceEnv_CircDiamond(self, env_S):  # env_S都是环境节点，寻找系统节点集合sys_S，每个属于sys_S的节点都存在选择某个动作后掉入env_S
        T = set()
        for env_node in env_S:  # 对每个env_S中的节点找其前驱，即系统节点，判断该系统节点是否存在选择某种动作后一定到达env_S
            if env_node not in self.reverse_undergraph.nodes: continue
            for sys_node in self.reverse_undergraph.nodes[env_node]:  #
                assert sys_node.kind == 'sys'
                ExitsActionIsOK = False
                for k in sys_node.actions:
                    ActionIsOK = True
                    for out in sys_node.actions[k]:
                        assert out[0].kind == 'env'
                        if out[0] not in env_S:
                            ActionIsOK = False
                            break
                    if ActionIsOK:
                        ExitsActionIsOK = True
                        break
                if ExitsActionIsOK:
                    T.add(sys_node)
        return T

    def _advanceSys_CircDiamond(self, sys_S):  # sys_S都是系统节点，寻找环境节点集合env_S，每个属于env_S的节点无论做动作选择都会掉入sys_S
        T = set()
        for sys_node in sys_S:
            for env_node in self.reverse_undergraph.nodes[sys_node]:
                assert env_node.kind == 'env'
                AllActionsAreOK = True
                for k in env_node.actions:
                    ActionIsOK = True
                    for out in env_node.actions[k]:
                        assert out[0].kind == 'sys'
                        if out[0] not in sys_S:
                            ActionIsOK = False
                            break
                    if not ActionIsOK:
                        AllActionsAreOK = False
                        break
                if AllActionsAreOK:
                    T.add(env_node)
        return T

    '''
    算子circdiamond
    '''

    def advance_CircDiamond(self, S):  # 博弈结构上的mu算子: mu_op S，即在一步中，对于环境的任意走法，系统都能使其落在集合S中
        sys_S = set()
        env_S = set()
        for node in S:
            if node.kind == 'sys':
                sys_S.add(node)
            else:
                env_S.add(node)
        return self._advanceEnv_CircDiamond(env_S) | self._advanceSys_CircDiamond(sys_S)

    '''
    F B = mu Y.( O Y or B )
    '''

    def F_CircDiamond(self, B):
        iterY = FixPoint()
        Y = set()
        while iterY.advance(Y):
            Y = self.advance_CircDiamond(Y) | B
        return Y

    '''
    G B = nu Y.( O Y and B )
    '''

    def G_CircDiamond(self, B):
        iterY = FixPoint()
        Y = copy.copy(self.MdpNodes)
        while iterY.advance(Y):
            Y = self.advance_CircDiamond(Y) & B
        return Y

    ''' 
    '''

    def _advanceSys_CircBox(self, sys_S):
        T = set()
        for sys_node in sys_S:
            for env_node in self.reverse_undergraph.nodes[sys_node]:
                assert env_node.kind == 'env'
                ExitsActionIsOK = False
                for k in env_node.actions:
                    ActionIsOK = True
                    for out in env_node.actions[k]:
                        assert out[0].kind == 'sys'
                        if out[0] not in sys_S:
                            ActionIsOK = False
                            break
                    if ActionIsOK:
                        ExitsActionIsOK = True
                        break
                if ExitsActionIsOK:
                    T.add(env_node)
        return T

    '''
    '''

    def _advanceEnv_CircBox(self, env_S):
        T = set()
        for env_node in env_S:
            if env_node not in self.reverse_undergraph.nodes: continue
            for sys_node in self.reverse_undergraph.nodes[env_node]:
                assert sys_node.kind == 'sys'
                AllActionsAreOK = True
                for k in sys_node.actions:
                    ActionIsOK = True
                    for out in sys_node.actions[k]:
                        assert out[0].kind == 'env'
                        if out[0] not in env_S:
                            ActionIsOK = False
                            break
                    if not ActionIsOK:
                        AllActionsAreOK = False
                        break
                if AllActionsAreOK:
                    T.add(sys_node)
        return T

    '''
    算子circbox
    '''

    def advance_CircBox(self, S):
        sys_S = set()
        env_S = set()
        for node in S:
            if node.kind == 'sys':
                sys_S.add(node)
            else:
                env_S.add(node)
        return self._advanceSys_CircBox(sys_S) | self._advanceEnv_CircBox(env_S)

    '''
    G B = nu Y.( O Y and B )
    '''

    def G_CircBox(self, B):
        iterY = FixPoint()
        Y = copy.copy(self.MdpNodes)
        while iterY.advance(Y):
            Y = self.advance_CircBox(Y) & B
        return Y

    '''
    @msg: 核心函数之一，计算博弈上到达B的概率
    '''

    def F_pr(self, B):
        dicB = {}
        for node in B:
            dicB[node] = 1
        # 一定能到达B的集合D, Pr(D)=1
        D = self.F_CircDiamond(set(dicB))
        for node in D:
            dicB[node] = 1

        # 一定不能到达D的集合C, Pr(C)=0
        C = self.G_CircBox(self.MdpNodes - set(dicB))
        for node in C:
            dicB[node] = 0

        # 迭代法用于验证
        val0 = {}
        val1 = {}
        for node in self.MdpNodes:
            if node in dicB:
                val0[node] = dicB[node]
                val1[node] = dicB[node]
            else:
                val0[node] = 0
                val1[node] = 1

        print('迭代法求解最小不动点')
        # 注意，这里的low是一个字典，它的键是MdpNode类型
        low = self.method_iter(val0, dicB)
        # 将low转成正常的字典，即键为字符串，方便获取值
        for k, v in low.items():
            self.pr[k.label] = v
        print(low)

        # print('迭代法求解最大不动点')
        up = self.method_iter(val1, dicB)
        # print(up)

        # print('sat求解')
        ret_sat = self.method_sat(dicB, up, low)
        return ret_sat

    '''
    @msg: 核心函数之一，计算博弈上能够无限多次经过B的概率
    '''

    def GF_pr(self, B):
        return self.F_pr(self.AGEF(B))

    '''
    @msg: CTL算子 AX
    '''

    def AX(self, B):
        T = set()
        for u in B:
            if u not in self.reverse_undergraph.nodes: continue
            for v in self.reverse_undergraph.nodes[u]:
                vIsOk = True
                for k in v.actions:
                    kIsOk = True
                    for out in v.actions[k]:
                        if out[0] not in B:
                            kIsOk = False
                            break
                    if kIsOk is False:
                        vIsOk = False
                        break
                if vIsOk is True:
                    T.add(v)
        return T

    '''
    @msg: CTL算子 EX
    '''

    def EX(self, B):
        T = set()
        for u in B:
            if u not in self.reverse_undergraph.nodes: continue
            for v in self.reverse_undergraph.nodes[u]:
                T.add(v)
        return T

    '''
    @msg: CTL上的 AG EF B = nu Z.[ mu Y.( EX Y or B ) and AX Z ]
    '''

    def AGEF(self, B):
        iterZ = FixPoint()
        Z = copy.copy(self.MdpNodes)
        while iterZ.advance(Z):
            Y = set()
            iterY = FixPoint()
            while iterY.advance(Y):
                Y = self.EX(Y) | B
            Z = Y & self.AX(Z)
        return Z

    '''
    @msg: Pr(s|=FG B) = F(G_circDiamond B)
    '''

    def FG_pr(self, B):
        return self.F_pr(self.G_CircDiamond(B))

    '''
    @msg: GR(1)接受条件 '/\GF B_i -> /\GF B_j'
    '''

    def GRoneAcc(self, set_Be, set_Bs):
        T = copy.copy(self.MdpNodes)
        for B_j in set_Bs:
            T = T & self.AGEF(B_j)
        print(T)
        C = set()
        for i in self.MdpNodes:
            if i.label == 'sysFail' or i.label == 'envSucc':
                C.add(i)
        T2 = set()
        for B_i in set_Be:
            T2 |= self.G_CircDiamond(self.MdpNodes - B_i - C)
        print(T2)
        return self.F_pr(T | T2)

    def getB(self, B):
        ret = set()
        for node in self.MdpNodes:
            if node.label in B:
                ret.add(node)
        return ret


def test1():
    m = Mdp()

    # m.__str__()可以返回MdpNodes中节点的相关信息
    # print(m.__str__())  # 此时MdpNodes中还没有内容

    # 问：ReadMdpFromFile这个函数里到底发生了什么？
    # 答：这个函数中调用了_addEdge函数和CompleteMdp函数，这两个函数都是先把图的节点和边定义好，但并没有真正画图
    m.ReadMdpFromFile('mdp')

    # print(m.__str__())  # 现在MdpNodes中有内容了

    m.dot(set())  # 这个函数很关键，没有它就不显示图，再下面的内容就是计算概率了

    B = m.getB(['s7'])
    m.F_pr(B)

    # 动作信息：
    # for node in m.MdpNodes:
    #     for k in node.actions.keys():
    #         print("node.label:", node.label)
    #         print("node.actions[k]:", node.actions[k])
    #         print("nod.actions.keys:", k)


def test2():
    m = Mdp()
    m.ReadMdpFromFile('mdp')
    m.dot(set())
    # 人为设定集合B中的元素
    B = m.getB(['s7'])
    m.F_pr(B)


if __name__ == "__main__":
    test1()
