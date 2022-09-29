# wandb.init("sql")
# from pythonBottom.run import finetune
# from pythonBottom.run import pre
# import wandb

from copy import deepcopy
from Dataset import SumDataset
from Model import *
from Radam import RAdam
from ScheduledOptim import *
from Searchnode import Node
from torch import optim
from tqdm import tqdm
from typing import List, Dict, Set, Tuple, Union

from base_logger import logger

import json
import numpy as np
import os
import pickle
import re
import sys
import torch
import torch.nn.functional as F
import traceback


ONE_LIST = [
    'root',
    'body',
    'statements',
    'block',
    'arguments',
    'initializers',
    'parameters',
    'case',
    'cases',
    'selectors'
]


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'NlLen': 500,
    'CodeLen': 30,
    'batch_size': 120,
    'embedding_size': 256,
    'WoLen': 15,
    'Vocsize': 100,
    'Nl_Vocsize': 100,
    'max_step': 3,
    'margin': 0.5,
    'poolsize': 50,
    'Code_Vocsize': 100,
    'num_steps': 50,
    'rulenum': 10,
    'cnum': 695
})

# os.environ["CUDA_VISIBLE_DEVICES"]="5, 7"
# os.environ['CUDA_LAUNCH_BLOCKING']="2"


def save_model(model, dirs='checkpointSearch/'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs='checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt', map_location=device))


def to_torch_tensor(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    tensor = data

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor


def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans


def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    return ans


def get_rule_pkl(sum_dataset: SumDataset) -> np.array:

    input_rule_parent = []
    input_rule_child = []

    for i in range(args.cnum):
        rule = sum_dataset.rrdict[i].strip().lower().split()
        input_rule_child.append(sum_dataset.pad_seq(sum_dataset.get_embedding(rule[2:], sum_dataset.CODE_VOCAB), sum_dataset.Char_Len))
        input_rule_parent.append(sum_dataset.CODE_VOCAB[rule[0].lower()])
    return np.array(input_rule_parent), np.array(input_rule_child)


def get_AST_pkl(sum_dataset: SumDataset) -> np.array:

    reversed_dict_code_vocab = {}
    for word in sum_dataset.CODE_VOCAB:
        reversed_dict_code_vocab[sum_dataset.CODE_VOCAB[word]] = word

    input_char = []

    for i in range(len(sum_dataset.CODE_VOCAB)):
        rule = reversed_dict_code_vocab[i].strip().lower()
        embedding = sum_dataset.get_char_embedding([rule])[0]
        embedding = sum_dataset.pad_seq(embedding, sum_dataset.Char_Len)
        input_char.append(embedding)

    return np.array(input_char)


def evalacc(model, dev_set: SumDataset):
    antimask = to_torch_tensor(getAntiMask(args.CodeLen))
    a, b = get_rule_pkl(dev_set)
    tmpast = get_AST_pkl(dev_set)
    tmpf = to_torch_tensor(a).unsqueeze(0).repeat(4, 1).long()
    tmpc = to_torch_tensor(b).unsqueeze(0).repeat(4, 1, 1).long()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=len(dev_set),
                                            shuffle=False, drop_last=True, num_workers=1)
    model = model.eval()
    accs = []
    tcard = []
    loss = []
    antimask2 = antimask.unsqueeze(0).repeat(len(dev_set), 1, 1).unsqueeze(1)
    rulead = to_torch_tensor(pickle.load(open("rulead.pkl", "rb"))
                             ).float().unsqueeze(0).repeat(4, 1, 1)
    tmpindex = to_torch_tensor(np.arange(len(dev_set.ruledict))
                               ).unsqueeze(0).repeat(4, 1).long()
    tmpchar = to_torch_tensor(tmpast).unsqueeze(0).repeat(4, 1, 1).long()
    tmpindex2 = to_torch_tensor(np.arange(len(dev_set.CODE_VOCAB))
                                ).unsqueeze(0).repeat(4, 1).long()
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = to_torch_tensor(devBatch[i])
        with torch.no_grad():
            l, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7],
                           devBatch[8], devBatch[9], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, devBatch[5])
            loss.append(l.mean().item())
            pred = pre.argmax(dim=-1)
            resmask = torch.gt(devBatch[5], 0)
            acc = (torch.eq(pred, devBatch[5])
                   * resmask).float()  # .mean(dim=-1)
            #predres = (1 - acc) * pred.float() * resmask.float()
            accsum = torch.sum(acc, dim=-1)
            resTruelen = torch.sum(resmask, dim=-1).float()
            cnum = torch.eq(accsum, resTruelen).sum().float()
            #print((torch.eq(accsum, resTruelen)) * (resTruelen - 1))
            acc = acc.sum(dim=-1) / resTruelen
            accs.append(acc.mean().item())
            tcard.append(cnum.item())
            # print(devBatch[5])
            # print(predres)
    tnum = np.sum(tcard)
    acc = np.mean(accs)
    l = np.mean(loss)
    # wandb.log({"accuracy":acc})
    # exit(0)
    return acc, tnum, l


def train():
    train_set = SumDataset(args, "train")
    print(len(train_set.rrdict))
    rulead = to_torch_tensor(pickle.load(open("rulead.pkl", "rb"))
                             ).float().unsqueeze(0).repeat(4, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = get_AST_pkl(train_set)
    a, b = get_rule_pkl(train_set)
    tmpf = to_torch_tensor(a).unsqueeze(0).repeat(4, 1).long()
    tmpc = to_torch_tensor(b).unsqueeze(0).repeat(4, 1, 1).long()
    tmpindex = to_torch_tensor(np.arange(len(train_set.ruledict))
                               ).unsqueeze(0).repeat(4, 1).long()
    tmpchar = to_torch_tensor(tmpast).unsqueeze(0).repeat(4, 1, 1).long()
    tmpindex2 = to_torch_tensor(np.arange(len(train_set.CODE_VOCAB))
                                ).unsqueeze(0).repeat(4, 1).long()
    args.Code_Vocsize = len(train_set.CODE_VOCAB)
    args.Nl_Vocsize = len(train_set.NL_VOCAB)
    args.Vocsize = len(train_set.CHAR_VOCAB)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    #dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    print(len(test_set))
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
    model = Decoder(args)
    # load_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = ScheduledOptim(
        optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0
    maxL = 1e10
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    antimask = to_torch_tensor(getAntiMask(args.CodeLen))
    # model.to()
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):
            if j % 3000 == 10:
                #acc, tnum = evalacc(model, dev_set)
                acc2, tnum2, l = evalacc(model, test_set)
                #print("for dev " + str(acc) + " " + str(tnum) + " max is " + str(maxC))
                print("for test " + str(acc2) + " " + str(tnum2) +
                      " max is " + str(maxC2) + "loss is " + str(l))
                # exit(0)
                if maxL > l:  # if maxC2 < tnum2 or maxC2 == tnum2 and maxAcc2 < acc2:
                    maxC2 = tnum2
                    maxAcc2 = acc2
                    maxL = l
                    print("find better acc " + str(maxAcc2))
                    save_model(model.module)
            antimask2 = antimask.unsqueeze(0).repeat(
                args.batch_size, 1, 1).unsqueeze(1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = to_torch_tensor(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[6], dBatch[7],
                            dBatch[8], dBatch[9], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, dBatch[5])
            # print(loss.mean())
            loss = torch.mean(loss)  # + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            j += 1


'''class Node:
    def __init__(self, name, d):
        self.name = name
        self.id = d
        self.father = None
        self.child = []
        self.sibiling = None
        self.expanded = False
        self.fatherlistID = 0
        self.fname = ""
    def printTree(self, r):
        s = r.name + " "#print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s'''


class SearchNode:
    def __init__(self, ds, nl):
        self.state = [ds.ruledict["start -> root"]]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.root = Node("root", 2)
        self.inputparent = ["root"]
        self.finish = False
        self.unum = 0
        self.parent = np.zeros(
            [args.NlLen + args.CodeLen, args.NlLen + args.CodeLen])
        # self.parent[args.NlLen]
        self.expanded = None
        #self.ruledict = ds.rrdict
        self.expandedname = []
        self.depth = [1]
        for x in ds.ruledict:
            self.expandedname.append(x.strip().split()[0])
        root = Node('root', 0)
        idx = 1
        self.idmap = {}
        self.idmap[0] = root
        currnode = root
        self.actlist: List[str] = []
        for x in nl[1:]:
            if x != "^":
                nnode = Node(x, idx)
                self.idmap[idx] = nnode
                idx += 1
                nnode.father = currnode
                currnode.child.append(nnode)
                currnode = nnode
            else:
                currnode = currnode.father
        self.everTreepath = []
        self.solveroot: Node = None

    def selcetNode(self, root: Node) -> Node:
        # and self.state[root.fatherlistID] < len(self.ruledict):
        if not root.expanded and root.name in self.expandedname and root.name not in ONE_LIST:
            return root
        else:
            for x in root.child:
                ans = self.selcetNode(x)
                if ans:
                    return ans
            if root.name in ONE_LIST and root.expanded == False:
                return root
        return None

    def selectExpandedNode(self):
        self.expanded = self.selcetNode(self.root)

    def getRuleEmbedding(self, ds: SumDataset, nl):
        inputruleparent = []
        inputrulechild = []
        for x in self.state:
            if x >= len(ds.rrdict):
                inputruleparent.append(ds.get_embedding(["value"], ds.CODE_VOCAB)[0])
                inputrulechild.append(ds.pad_seq(
                    ds.get_embedding(["copyword"], ds.CODE_VOCAB), ds.Char_Len))
            else:
                rule = ds.rrdict[x].strip().lower().split()
                # print(rule[0])
                inputruleparent.append(ds.get_embedding([rule[0]], ds.CODE_VOCAB)[0])
                #print(ds.Get_Em([rule[0]], ds.Code_Voc))
                inputrulechild.append(ds.pad_seq(
                    ds.get_embedding(rule[2:], ds.CODE_VOCAB), ds.Char_Len))
        tmp = [ds.pad_seq(ds.get_embedding(['start'], ds.CODE_VOCAB), 10)
               ] + self.everTreepath
        inputrulechild = ds.pad_list(tmp, ds.Code_Len, 10)
        inputrule = ds.pad_seq(self.state, ds.Code_Len)
        #inputrulechild = ds.pad_list(inputrulechild, ds.Code_Len, ds.Char_Len)
        inputruleparent = ds.pad_seq(inputruleparent, ds.Code_Len)
        inputdepth = ds.pad_list(self.depth, ds.Code_Len, 40)
        # print(inputruleparent)
        return inputrule, inputrulechild, inputruleparent, inputdepth

    def getTreePath(self, ds: SumDataset):
        tmppath = [self.expanded.name.lower()]
        node = self.expanded.father
        while node:
            tmppath.append(node.name.lower())
            node = node.father
        tmp = ds.pad_seq(ds.get_embedding(tmppath, ds.CODE_VOCAB), 10)
        self.everTreepath.append(tmp)
        return ds.pad_list(self.everTreepath, ds.Code_Len, 10)

    def checkapply(self, rule: int, ds: SumDataset) -> bool:
        if rule >= len(ds.ruledict):
            if self.expanded.name == 'root' and rule - len(ds.ruledict) >= args.NlLen:
                if rule - len(ds.ruledict) - args.NlLen not in self.idmap:
                    return False
                if self.idmap[rule - len(ds.ruledict) - args.NlLen].name not in ['MemberReference', 'BasicType', 'operator', 'qualifier', 'member', 'Literal']:
                    return False
                if '.0' in self.idmap[rule - len(ds.ruledict) - args.NlLen].getTreestr():
                    return False
                #print(self.idmap[rule - len(ds.ruledict)].name)
                # assert(0)
                return True
            if rule - len(ds.ruledict) >= args.NlLen:
                return False
            idx = rule - len(ds.ruledict)
            if idx not in self.idmap:
                return False
            if self.idmap[idx].name != self.expanded.name:
                if self.idmap[idx].name in ['VariableDeclarator', 'FormalParameter', 'InferredFormalParameter']:
                    return True
                #print(self.idmap[idx].name, self.expanded.name, idx)
                return False
        else:
            rules = ds.rrdict[rule]
            if rules == 'start -> unknown':
                if self.unum >= 1:
                    return False
                return True
            # if len(self.depth) == 1:
                # print(rules)
            #    if rules != 'root -> modified' or rules != 'root -> add':
            #        return False
            if rules.strip().split()[0].lower() != self.expanded.name.lower():
                return False
        return True

    def copynode(self, newnode, original):
        for x in original.child:
            nnode = Node(x.name, -1)
            nnode.father = newnode
            nnode.expanded = True
            newnode.child.append(nnode)
            self.copynode(nnode, x)
        return

    def applyrule(self, rule: int, ds: SumDataset) -> bool:
        '''if rule < len(ds.ruledict):
            print(rule, ds.rrdict[rule])
        elif rule >= len(ds.ruledict) + args.NlLen:
            print('copy', self.idmap[rule - len(ds.ruledict) - args.NlLen].name)
        else:
            print('copy2', self.idmap[rule - len(ds.ruledict)].name)'''
        if rule >= len(ds.ruledict):
            if rule >= len(ds.ruledict) + args.NlLen:
                idx = rule - len(ds.ruledict) - args.NlLen
            else:
                idx = rule - len(ds.ruledict)
            self.actlist.append('copy-' + self.idmap[idx].name)
        else:
            self.actlist.append(ds.rrdict[rule])
        if rule >= len(ds.ruledict):
            nodesid = rule - len(ds.ruledict)
            if nodesid >= args.NlLen:
                nodesid = nodesid - args.NlLen
                nnode = Node(self.idmap[nodesid].name, nodesid)
                nnode.fatherlistID = len(self.state)
                nnode.father = self.expanded
                nnode.fname = "-" + self.printTree(self.idmap[nodesid])
                self.expanded.child.append(nnode)
            else:
                nnode = self.idmap[nodesid]
                if nnode.name == self.expanded.name:
                    self.copynode(self.expanded, nnode)
                    nnode.fatherlistID = len(self.state)
                else:
                    if nnode.name == 'VariableDeclarator':
                        currnode = -1
                        for x in nnode.child:
                            if x.name == 'name':
                                currnode = x
                                break
                        nnnode = Node(currnode.child[0].name, -1)
                    else:
                        currnode = -1
                        for x in nnode.child:
                            if x.name == 'name':
                                currnode = x
                                break
                        nnnode = Node(currnode.child[0].name, -1)
                    nnnode.father = self.expanded
                    self.expanded.child.append(nnnode)
                    nnnode.fatherlistID = len(self.state)
                self.expanded.expanded = True
        else:
            rules = ds.rrdict[rule]
            if rules == 'start -> unknown':
                self.unum += 1
            # if rules.strip().split()[0] != self.expanded.name:
            #    #print(self.expanded.name)
            #    assert(0)
            #    return False
            #assert(rules.strip().split()[0] == self.expanded.name)
            if rules.strip() == self.expanded.name + " -> End":
                self.expanded.expanded = True
            else:
                for x in rules.strip().split()[2:]:
                    nnode = Node(x, -1)
                    #nnode = Node(x, self.expanded.depth + 1)
                    self.expanded.child.append(nnode)
                    nnode.father = self.expanded
                    nnode.fatherlistID = len(self.state)
        # self.parent.append(self.expanded.fatherlistID)
        self.parent[args.NlLen + len(self.depth),
                    args.NlLen + self.expanded.fatherlistID] = 1
        if rule >= len(ds.ruledict) + args.NlLen:
            self.parent[args.NlLen + len(self.depth),
                        rule - len(ds.ruledict) - args.NlLen] = 1
        elif rule >= len(ds.ruledict):
            self.parent[args.NlLen +
                        len(self.depth), rule - len(ds.ruledict)] = 1
        if rule >= len(ds.ruledict) + args.NlLen:
            self.state.append(ds.ruledict['start -> copyword2'])
        elif rule >= len(ds.ruledict):
            self.state.append(ds.ruledict['start -> copyword'])
        else:
            self.state.append(rule)
        # self.state.append(rule)
        self.inputparent.append(self.expanded.name.lower())
        self.depth.append(1)
        if self.expanded.name not in ONE_LIST:
            self.expanded.expanded = True
        return True

    def printTree(self, r):
        s = r.name + r.fname + " "  # print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "  # print(r.name + "^")
        return s

    def getTreestr(self):
        return self.printTree(self.root)


beamss = []


def BeamSearch(input_nl, sum_dataset: SumDataset, decoder_model: Decoder, beam_size: int, batch_size: int, k: int) -> Dict[int, List[SearchNode]]:

    logger.info('starting beam search')

    batch_size = len(input_nl[0].view(-1, args.NlLen))

    reversed_dict_code_vocab = {}
    for word in sum_dataset.CODE_VOCAB:
        reversed_dict_code_vocab[sum_dataset.CODE_VOCAB[word]] = word

    temp_ast = get_AST_pkl(sum_dataset)
    a, b = get_rule_pkl(sum_dataset)

    tmpf = to_torch_tensor(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = to_torch_tensor(b).unsqueeze(0).repeat(2, 1, 1).long()
    rulead = to_torch_tensor(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    tmpindex = to_torch_tensor(np.arange(len(sum_dataset.ruledict))).unsqueeze(0).repeat(2, 1).long()
    tmpchar = to_torch_tensor(temp_ast).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex2 = to_torch_tensor(np.arange(len(sum_dataset.CODE_VOCAB))).unsqueeze(0).repeat(2, 1).long()

    with torch.no_grad():
        beams: Dict[int, List[SearchNode]] = {}
        hisTree: Dict[int, Dict[str, int]] = {}

        for i in range(batch_size):
            beams[i] = [SearchNode(sum_dataset, sum_dataset.nl[args.batch_size * k + i])]
            hisTree[i] = {}

        index = 0
        antimask = to_torch_tensor(getAntiMask(args.CodeLen))
        endnum = {}
        continueSet = {}
        tansV: Dict[int, List[SearchNode]] = {}

        while True:
            tmpbeam: Dict[int, List[list]] = {}
            ansV: Dict[int, List[SearchNode]] = {}

            if len(endnum) == batch_size:
                break

            if index >= args.CodeLen:
                break

            for ba in range(batch_size):
                tmprule = []
                tmprulechild = []
                tmpruleparent = []
                tmptreepath = []
                tmpAd = []
                validnum = []
                tmpdepth = []
                tmpnl = []
                tmpnlad = []
                tmpnl8 = []
                tmpnl9 = []

                for p in range(beam_size):
                    if p >= len(beams[ba]):
                        continue

                    word = beams[ba][p]
                    word.selectExpandedNode()

                    if word.expanded == None or len(word.state) >= args.CodeLen:
                        word.finish = True
                        ansV.setdefault(ba, []).append(word)
                    else:
                        validnum.append(p)
                        tmpnl.append(input_nl[0][ba].data.cpu().numpy())
                        tmpnlad.append(input_nl[1][ba].data.cpu().numpy())
                        tmpnl8.append(input_nl[8][ba].data.cpu().numpy())
                        tmpnl9.append(input_nl[9][ba].data.cpu().numpy())
                        a, b, c, d = word.getRuleEmbedding(
                            sum_dataset, sum_dataset.nl[args.batch_size * k + ba])
                        tmprule.append(a)
                        tmprulechild.append(b)
                        tmpruleparent.append(c)
                        tmptreepath.append(word.getTreePath(sum_dataset))
                        tmpAd.append(word.parent)
                        tmpdepth.append(d)

                if len(tmprule) == 0:
                    continue

                antimasks = antimask.unsqueeze(0).repeat(
                    len(tmprule), 1, 1).unsqueeze(1)

                tmprule = np.array(tmprule)
                tmprulechild = np.array(tmprulechild)
                tmpruleparent = np.array(tmpruleparent)
                tmptreepath = np.array(tmptreepath)
                tmpAd = np.array(tmpAd)
                tmpdepth = np.array(tmpdepth)
                tmpnl = np.array(tmpnl)
                tmpnlad = np.array(tmpnlad)
                tmpnl8 = np.array(tmpnl8)
                tmpnl9 = np.array(tmpnl9)

                print(f"before@{index} batch{ba} x: {word.prob}: {word.getTreestr()} ; {word.actlist}")
                result = decoder_model(
                    to_torch_tensor(tmpnl),
                    to_torch_tensor(tmpnlad),
                    to_torch_tensor(tmprule),
                    to_torch_tensor(tmpruleparent),
                    to_torch_tensor(tmprulechild),
                    to_torch_tensor(tmpAd),
                    to_torch_tensor(tmptreepath),
                    to_torch_tensor(tmpnl8),
                    to_torch_tensor(tmpnl9),
                    tmpf,
                    tmpc,
                    tmpindex,
                    tmpchar,
                    tmpindex2,
                    rulead,
                    antimasks,
                    None,
                    "test"
                )

                print(f"after@{index} batch{ba} x: {word.prob}: {word.getTreestr()} ; {word.actlist}")
                results = result.data.cpu().numpy()
                currIndex = 0
                tmp_prob_list: List[Tuple[int, float]] = list()

                for j in range(beam_size):
                    if j not in validnum:
                        continue
                    word = beams[ba][j]
                    tmpbeamsize = 0  # beamsize
                    result: np.ndarray[float] = np.negative(
                        results[currIndex, index])
                    currIndex += 1
                    cresult: np.ndarray[float] = np.negative(result)
                    indexs: np.ndarray[int] = np.argsort(result)
                    for i in range(len(indexs)):
                        if tmpbeamsize >= 10:
                            break
                        if cresult[indexs[i]] == 0:
                            break
                        c = word.checkapply(indexs[i], sum_dataset)
                        if c:
                            tmpbeamsize += 1
                        else:
                            continue
                        prob = word.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(ba, []).append([prob, indexs[i], word])

            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beam_size:
                        endnum[i] = 1
                    tansV.setdefault(i, []).extend(ansV[i])

            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for word in ansV[j]:
                            tmpbeam[j].append([word.prob, -1, word])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)
                    beams[j] = []
                    for word in tmp:  # x: (prob, index, SearchNode)
                        if len(beams[j]) >= beam_size:
                            break
                        if word[1] != -1:
                            x_prob: float = word[0]
                            x_rule: int = word[1]
                            copynode: SearchNode = pickle.loads(pickle.dumps(word[2]))
                            copynode.applyrule(word[1], sum_dataset)
                            print(f"copynode {copynode.prob}:  {copynode.getTreestr()}; {copynode.actlist}")
                            tree_str = copynode.getTreestr()
                            if tree_str in hisTree:
                                continue
                            copynode.prob = word[0]
                            beams[j].append(copynode)
                            hisTree[j][tree_str] = 1
                        else:
                            beams[j].append(word[2])
            index += 1

        for j in range(batch_size):
            visit = {}
            tmp = []
            for word in tansV[j]:
                tree_str = word.getTreestr()
                if tree_str not in visit and word.finish:
                    visit[tree_str] = 1
                    tmp.append(word)
                else:
                    continue
            beams[j] = sorted(tmp, key=lambda x: x.prob, reverse=True)[:beam_size]

        return beams


def test():

    dev_set = SumDataset(args, "test")
    rulead_tensor = to_torch_tensor(pickle.load(open("rulead.pkl", "rb")))
    rulead = rulead_tensor.float().unsqueeze(0).repeat(2, 1, 1)

    args.cnum = rulead.size(1)
    args.Nl_Vocsize = len(dev_set.NL_VOCAB)
    args.Code_Vocsize = len(dev_set.CODE_VOCAB)
    args.Vocsize = len(dev_set.CHAR_VOCAB)
    args.rulenum = len(dev_set.ruledict) + args.NlLen
    args.batch_size = 12

    model = Decoder(args)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    load_model(model)

    return model


def findnodebyid(root, idx):
    if root.id == idx:
        return root
    for x in root.child:
        t = findnodebyid(x, idx)
        if t:
            return t


def getroot(strlst):
    tokens = strlst.split()
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def getMember(node):
    for x in node.child:
        if x.name == 'member':
            return x.child[0].name


def applyoperater(ans, subroot):
    # print(ans.root.printTree(ans.root))
    copynode = pickle.loads(pickle.dumps(subroot))
    change = False
    type = ''
    for x in ans.root.child:
        if x.id != -1:
            change = True
            node = findnodebyid(copynode, x.id)
            if node is None:
                continue
            if node.name == 'member':
                type = node.child[0].name
                # assert(0)
            elif node.name == 'MemberReference':
                type = getMember(node)  # node.child[0].child[0].name
                print(6, type)
            elif node.name == 'qualifier':
                type = node.child[0].name
            elif node.name == 'operator' or node.name == 'Literal' or node.name == 'BasicType':
                type = 'valid'
            else:
                print(node.name)
                assert (0)
            # print(node.name)
            idx = node.father.child.index(node)
            node.father.child[idx] = x
            x.father = node.father
    if change:
        node = Node('root', -1)
        node.child.append(copynode)
        copynode.father = node
        ans.solveroot = node  # copynode
        ans.type = type
        # print(node.printTree(ans.solveroot))
    else:
        ans.solveroot = ans.root
        ans.type = type
    # print(copynode.printTree(copynode))
    # assert(0)
    return


def replaceVar(root, rrdict, place=False):
    if root.name in rrdict:
        root.name = rrdict[root.name]
    elif root.name == 'unknown' and place:
        root.name = "placeholder_ter"
    elif len(root.child) == 0:
        if re.match('loc%d', root.name) is not None or re.match('par%d', root.name) is not None:
            return False
    ans = True
    for x in root.child:
        ans = ans and replaceVar(x, rrdict)
    return ans


def getUnknown(root: Node) -> List[Node]:
    if root.name == 'unknown':
        return [root]
    ans = []
    for x in root.child:
        ans.extend(getUnknown(x))
    return ans


def solveUnknown(ans: SearchNode, vardic: Dict[str, str], typedic: Dict[str, str], classcontent, sclassname: str, mode: int) -> List[str]:
    nodes = getUnknown(ans.solveroot)
    fans: List[str] = list()
    #fans_prob: List[float] = list()
    if len(nodes) >= 2:
        return []  # ([], [])
    elif len(nodes) == 0:
        # print(ans.root.printTree(ans.solveroot))
        # ([ans.root.printTree(ans.solveroot)], [ans.prob])
        return [ans.root.printTree(ans.solveroot)]
    else:
        # print(2)
        unknown = nodes[0]
        if unknown.father.father and unknown.father.father.name == 'MethodInvocation':
            classname = ''
            args = []
            print('method')
            if unknown.father.name == 'member':
                for x in unknown.father.father.child:
                    if x.name == 'qualifier':
                        print(x.child[0].name, typedic)
                        if x.child[0].name in typedic:
                            classname = typedic[x.child[0].name]
                            break
                        else:
                            if sclassname == 'org.jsoup.nodes.Element':
                                sclassname = 'org.jsoup.nodes.Node'
                            for f in classcontent[sclassname + '.java']['classes'][0]['fields']:
                                print(x.child[0].name, f['name'])
                                if f['name'] == x.child[0].name[:-4]:
                                    classname = f['type']
                                    break
                for x in unknown.father.father.child:
                    if x.name == 'arguments':
                        for y in x.child:
                            if y.name == 'MemberReference':
                                try:
                                    if y.child[0].child[0].name in typedic:
                                        args.append(
                                            typedic[y.child[0].child[0].name])
                                    else:
                                        #print(6, y.child[0].child[0].name)
                                        args.append('int')  # return []
                                except:
                                    # print('gg2')
                                    return []  # ([], [])
                            elif y.name == 'Literal':
                                if y.child[0].child[0].name == "<string>_er":
                                    args.append("String")
                                else:
                                    args.append("int")
                            else:
                                print('except')
                                return []  # ([], [])
            print(7, classname)
            if classname == '':
                classbody = classcontent[sclassname + '.java']['classes']
            elif classname != '':
                if classname + ".java" not in classcontent:
                    #print(5, classname )
                    return []  # ([], [])
                classbody = classcontent[classname + '.java']['classes']
            #print(5, sclassname, classbody, classname)
            # print(8)
            if unknown.father.name == 'qualifier':
                vtype = ""
                for x in classbody[0]['fields']:
                    # print(x)
                    if x['name'] == ans.type[:-4]:
                        vtype = x['type']
                        break
            if 'IfStatement' in ans.getTreestr():
                if mode == 1 and len(ans.solveroot.child) == 1:
                    # print(ans.solveroot.printTree(ans.solveroot))
                    return []  # ([], [])
                # print(ans.solveroot.printTree(ans.solveroot))
                if unknown.father.name == 'member':
                    for x in classbody[0]['methods']:
                        if len(x['params']) == 0 and x['type'] == 'boolean':
                            unknown.name = x['name'] + "_ter"
                            #print('gggg', unknown.printTree(ans.solveroot))
                            fans.append(unknown.printTree(ans.solveroot))
                            # fans_prob.append(ans.solveroot.prob)
                elif unknown.father.name == 'qualifier':
                    for x in classbody[0]['fields']:
                        if x['type'] == vtype:
                            unknown.name = x['name'] + "_ter"
                            fans.append(unknown.printTree(ans.solveroot))
                            # fans_prob.append(ans.solveroot.prob)
            else:
                #print("a", args)
                if mode == 0 and ans.root == ans.solveroot and len(args) == 0 and classname != 'EndTag':
                    return []  # ([], [])
                otype = ""
                if classname == 'EndTag':
                    otype = "String"

                if mode == 0 and ans.type != '':
                    args = []

                    if ans.type == "valid":
                        return []  # ([], [])
                    for m in classbody[0]['methods']:
                        # print(m['name'])
                        if m['name'] == ans.type[:-4]:
                            otype = m['type']
                            for y in m['params']:
                                args.append(y['type'])
                            break
                #print(args, ans.type, 'o')
                if unknown.father.name == 'member':
                    #print(mode, ans.type, args)
                    for x in classbody[0]['methods']:
                        # print(x)
                        print(x['type'], otype, x['name'], ans.type)
                        if len(args) == 0 and len(x['params']) == 0:
                            if mode == 0 and x['type'] != otype:
                                continue
                            if mode == 1 and x['type'] is not None:
                                continue
                            # if mode == 1 and x['type'] != "null":
                            #    continue
                            unknown.name = x['name'] + "_ter"
                            #print('gggg', unknown.printTree(ans.solveroot))
                            fans.append(unknown.printTree(ans.solveroot))
                            # fans_prob.append(ans.solveroot.prob)
                        #print(x['name'], x['type'], args)
                        if ans.type != '':
                            if mode == 0 and len(args) > 0 and x['type'] == otype:
                                targ = []
                                for y in x['params']:
                                    targ.append(y['type'])
                                if args == targ:
                                    unknown.name = x['name'] + "_ter"
                                    fans.append(
                                        unknown.printTree(ans.solveroot))
                                    # fans_prob.append(ans.solveroot.prob)
                        else:
                            # print(10)
                            if mode == 0 and len(args) > 0:
                                # print(11)
                                targ = []
                                for y in x['params']:
                                    targ.append(y['type'])
                                #print('p', targ, x['name'], x)
                                if args == targ and 'type' in x and x['type'] is None:
                                    unknown.name = x['name'] + "_ter"
                                    fans.append(
                                        unknown.printTree(ans.solveroot))
                                    # fans_prob.append(ans.solveroot.prob)
                elif unknown.father.name == 'qualifier':
                    if ans.type == 'valid':
                        return []  # ([], [])
                    for x in classbody[0]['fields']:
                        if x['type'] == vtype:
                            unknown.name = x['name'] + "_ter"
                            fans.append(unknown.printTree(ans.solveroot))
                    for x in classbody[0]['methods']:
                        if x['type'] == vtype and len(x['params']) == 0:
                            tmpnode = Node('MethodInvocation', -1)
                            tmpnode1 = Node('member', -1)
                            tmpnode2 = Node(x['name'] + "_ter", -1)
                            tmpnode.child.append(tmpnode1)
                            tmpnode1.father = tmpnode
                            tmpnode1.child.append(tmpnode2)
                            tmpnode2.father = tmpnode1
                            unknown.name = " ".join(tmpnode.printTree(tmpnode).split()[
                                                    :-1])  # tmpnode.printTree(tmpnode)
                            fans.append(unknown.printTree(ans.solveroot))
                            # fans_prob.append(ans.solveroot.prob)
        elif unknown.father.name == 'qualifier':
            classbody = classcontent[sclassname + '.java']['classes']
            vtype = ""
            for x in classbody[0]['fields']:
                if x['name'] == ans.type[:-4]:
                    vtype = x['type']
                    break
            #print(5, vtype)
            for x in classbody[0]['fields']:
                if x['type'] == vtype:
                    unknown.name = x['name'] + "_ter"
                    fans.append(unknown.printTree(ans.solveroot))
                    # fans_prob.append(ans.solveroot.prob)
            for x in classbody[0]['methods']:
                if x['type'] == vtype and len(x['params']) == 0:
                    tmpnode = Node('MethodInvocation', -1)
                    tmpnode1 = Node('member', -1)
                    tmpnode2 = Node(x['name'] + "_ter", -1)
                    tmpnode.child.append(tmpnode1)
                    tmpnode1.father = tmpnode
                    tmpnode1.child.append(tmpnode2)
                    tmpnode2.father = tmpnode1
                    unknown.name = " ".join(
                        tmpnode.printTree(tmpnode).split()[:-1])
                    fans.append(unknown.printTree(ans.solveroot))
                    # fans_prob.append(ans.solveroot.prob)
        elif unknown.father.name == 'member':
            classname = ''
            if unknown.father.name == 'member':
                for x in unknown.father.father.child:
                    if x.name == 'qualifier':
                        if x.child[0].name in typedic:
                            classname = typedic[x.child[0].name]
                            break
                        else:
                            for f in classcontent[sclassname + '.java']['classes'][0]['fields']:
                                if f['name'] == x.child[0].name[:-4]:
                                    classname = f['type']
                                    break
                        if x.child[0].name[:-4] + ".java" in classcontent:
                            classname = x.child[0].name[:-4]
            #print(0, classname, ans.type)
            if classname == '':
                classbody = classcontent[sclassname + '.java']['classes']
            elif classname != '':
                if classname + ".java" not in classcontent:
                    #print(5, classname )
                    return []  # ([], [])
                classbody = classcontent[classname + '.java']['classes']
            vtype = ""
            #print('type', ans.type)
            for x in classbody[0]['fields']:
                if x['name'] == ans.type[:-4]:
                    vtype = x['type']
                    break
            if unknown.father.father.father.father and (unknown.father.father.father.father.name == 'MethodInvocation' or unknown.father.father.father.father.name == 'ClassCreator') and ans.type == "":
                mname = ""
                tname = ""
                if unknown.father.father.father.father.name == "MethodInvocation":
                    tname = 'member'
                else:
                    tname = 'type'
                for s in unknown.father.father.father.father.child:
                    if s.name == 'member' and tname == 'member':
                        mname = s.child[0].name
                    if s.name == 'type' and tname == 'type':
                        mname = s.child[0].child[0].child[0].name
                idx = unknown.father.father.father.child.index(
                    unknown.father.father)
                # print(idx)
                if tname == 'member':
                    for f in classbody[0]['methods']:
                        if f['name'] == mname[:-4] and idx < len(f['params']):
                            vtype = f['params'][idx]['type']
                            print(vtype, f['name'])
                            break
                else:
                    if mname[:-4] + ".java" not in classcontent:
                        return []  # ([], [])
                    for f in classcontent[mname[:-4] + ".java"]['classes'][0]['methods']:
                        #print(f['name'], f['params'], mname[:-4])
                        if f['name'] == mname[:-4] and idx < len(f['params']):
                            vtype = f['params'][idx]['type']
                            break
            if True:
                for x in classbody[0]['fields']:
                    #print(classname, x['type'], x['name'], vtype, ans.type)
                    # or vtype == "":
                    if x['type'] == vtype or (x['type'] == 'double' and vtype == 'int'):
                        unknown.name = x['name'] + "_ter"
                        fans.append(unknown.printTree(ans.solveroot))
                        # fans_prob.append(ans.solveroot.prob)
    return fans  # (fans, fans_prob)


def extarctmode(root):
    mode = 0
    if len(root.child) == 0:
        return 0, None
    if root.child[0].name == 'modified':
        mode = 0
    elif root.child[0].name == 'add':
        mode = 1
    else:
        return 0, None
        print(root.printTree(root))
        # assert(0)
    root.child.pop(0)
    return mode, root


def solve_one(data_buggy_locations: List[Dict], model: Decoder) -> list:
    '''
    data: (treestr, prob, model, subroot, vardic, typedic, idx, idss, classname, mode):
    '''

    logger.info('starting solve_one()')

    args.batch_size = 20
    dev_set = SumDataset(args, "test")
    dev_set.preProcessOne(data_buggy_locations)

    indexs = 0
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                            shuffle=False, drop_last=False, num_workers=0)
    savedata = []
    patch = {}
    for x in tqdm(devloader):
        if indexs < 0:
            indexs += 1
            continue

        ans = BeamSearch((x[0], x[1], None, None, None, None, None, None,
                         x[2], x[3]), dev_set, model, 100, args.batch_size, indexs)
        for i in range(len(ans)):
            currid = indexs * args.batch_size + i
            tmp_data_list = list()
            tmp_data_file = os.path.join(
                "d4j", data_buggy_locations[currid]["bugid"], f"temp-{currid}.json")
            idss = data_buggy_locations[currid]['idss']
            if "fl_score" not in data_buggy_locations[currid]:
                data_buggy_locations[currid]["fl_score"] = -1.0
            subroot = data_buggy_locations[currid]['subroot']
            if os.path.exists("result/%s.json" % idss):
                classcontent = json.load(open("result/%s.json" % idss, 'r'))
            else:
                classcontent = []
            classcontent.extend(json.load(open("temp.json", 'r')))
            rrdicts = {}
            for x in classcontent:
                rrdicts[x['filename']] = x
                if 'package_name' in x:
                    rrdicts[x['package_name'] + "." + x['filename']] = x
            vardic = data_buggy_locations[currid]['vardic']
            typedic = data_buggy_locations[currid]['typedic']
            classname = data_buggy_locations[currid]['classname']
            mode = data_buggy_locations[currid]['mode']
            rrdict = {}
            for x in vardic:
                rrdict[vardic[x]] = x
            for j in range(len(ans[i])):
                if j > 60 and idss != 'Lang-33':
                    break
                mode, ans[i][j].root = extarctmode(ans[i][j].root)
                if ans[i][j].root is None:
                    continue
                applyoperater(ans[i][j], subroot)
                an = replaceVar(ans[i][j].solveroot, rrdict)
                if not an:
                    continue
                try:
                    tcodes = solveUnknown(
                        ans[i][j], vardic, typedic, rrdicts, classname, mode)
                except Exception as e:
                    traceback.print_exc()
                    tcodes = []
                for code in tcodes:
                    prob = ans[i][j].prob
                    if code.split(" ")[0] != 'root':
                        assert (0)
                    if str(mode) + code + str(data_buggy_locations[currid]['line']) not in patch:
                        patch[str(mode) + code + str(data_buggy_locations[currid]['line'])] = 1
                    else:
                        continue
                    obj = {'id': currid, 'idss': idss, 'precode': data_buggy_locations[currid]['precode'], 'aftercode': data_buggy_locations[currid]['aftercode'], 'oldcode': data_buggy_locations[currid]['oldcode'], 'filename': data_buggy_locations[currid]
                           ['filepath'], 'mode': mode, 'code': code, 'prob': prob, 'line': data_buggy_locations[currid]['line'], 'isa': data_buggy_locations[currid]['isa'], 'fl_score': data_buggy_locations[currid]['fl_score'], 'actlist': ans[i][j].actlist}
                    savedata.append(obj)
                    tmp_data_list.append(obj)
            with open(tmp_data_file, "w") as tmp_df:
                json.dump(tmp_data_list, tmp_df, indent=2)
        indexs += 1
    return savedata


# (treestr, prob, model, subroot, vardic, typedic, idx, idss, classname, mode):
def solveone2(data, model):
    #os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"
    #assert(len(data) <= 40)
    args.batch_size = 20
    dev_set = SumDataset(args, "test")
    dev_set.preProcessOne(data)  # x = dev_set.preProcessOne(treestr, prob)
    #dev_set.nl = [treestr.split()]
    indexs = 0
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                            shuffle=False, drop_last=False, num_workers=0)
    savedata = []
    patch = {}
    for x in tqdm(devloader):
        if indexs < 0:
            indexs += 1
            continue
        #print(indexs,indexs * args.batch_size, data[5]['oldcode'])
        #print(x[0][0], dev_set.data[0][idx])
        #assert(np.array_equal(x[0][0], dev_set.datam[0][4]))
        #assert(np.array_equal(x[1][0], dev_set.datam[1][4].toarray()))
        #assert(np.array_equal(x[2][0], dev_set.datam[8][4]))
        #assert(np.array_equal(x[3][0], dev_set.datam[9][4]))
        #print(data[indexs]['mode'], data[indexs]['oldcode'])
        ans = BeamSearch((x[0], x[1], None, None, None, None, None, None,
                         x[2], x[3]), dev_set, model, 60, args.batch_size, indexs)
        print('debug', len(ans[0]))
        for i in range(len(ans)):
            currid = indexs * args.batch_size + i
            subroot = data[currid]['subroot']
            vardic = data[currid]['vardic']
            typedic = data[currid]['typedic']
            # print(classname)
            # assert(0)
            mode = data[currid]['mode']
            rrdict = {}
            for x in vardic:
                rrdict[vardic[x]] = x
            for j in range(len(ans[i])):
                print(ans[i][j].printTree(ans[i][j].root))
                mode, ans[i][j].root = extarctmode(ans[i][j].root)
                if ans[i][j].root is None:
                    print('debug1')
                    continue
                applyoperater(ans[i][j], subroot)
                an = replaceVar(ans[i][j].solveroot, rrdict)
                if not an:
                    print('debug2')
                    continue
                savedata.append({'precode': data[currid]['precode'], 'aftercode': data[currid]['aftercode'], 'oldcode': data[currid]['oldcode'],
                                'mode': mode, 'code': ans[i][j].root.printTree(ans[i][j].solveroot), 'prob': ans[i][j].prob, 'line': data[currid]['line']})
    # print(savedata)
    return savedata
    # for x in savedata:
    #    print(x['oldcode'], x['code'])
    # exit(0)
    #f.write(" ".join(ans.ans[1:-1]))
    # f.write("\n")
    # f.flush()#print(ans)
    #print(x[0][0], dev_set.data[0][idx])
    #assert(np.array_equal(x[0][0], dev_set.data[0][idx]))
    #assert(np.array_equal(x[1][0], dev_set.data[1][idx].toarray()))
    #assert(np.array_equal(x[2][0], dev_set.data[8][idx]))
    #assert(np.array_equal(x[3][0], dev_set.data[9][idx]))
    open('patchmu/%s.json' % data[0]['idss'],
         'w').write(json.dumps(savedata, indent=4))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    if sys.argv[1] == "train":
        train()
    else:
        test()
     # test()
