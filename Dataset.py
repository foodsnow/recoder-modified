import json
import os
import pickle
import random
import re
import sys
from typing import Dict, List, Set, Tuple, Union

import h5py
import numpy as np
import torch
import torch.utils.data as data
from nltk import word_tokenize
from scipy import sparse
from tqdm import tqdm

from base_logger import logger
from parse_dataflow import GetFlow
from vocab import VocabEntry


sys.setrecursionlimit(500000000)


class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train"):

        logger.info('starting to initialize SumDataset')

        self.train_path = "train_process.txt"
        self.val_path = "dev_process.txt"  # "validD.txt"
        self.test_path = "test_process.txt"
        self.NL_VOCAB = {"pad": 0, "Unknown": 1}
        self.CODE_VOCAB = {"pad": 0, "Unknown": 1}
        self.CHAR_VOCAB = {"pad": 0, "Unknown": 1}
        self.Nl_Len: int = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50

        self.rule_dict: Dict[str, int] = pickle.load(open("data_rule.pkl", "rb"))
        self.rule_dict['start -> copyword2'] = len(self.rule_dict)
        self.rule_reverse_dict = {}
        for x in self.rule_dict:
            self.rule_reverse_dict[self.rule_dict[x]] = x

        if not os.path.exists("data_nl_voc.pkl"):
            self.init_dic()

        self.Load_Voc()
        if dataName == "train":
            if os.path.exists("data_train_data.pkl"):
                self.data = pickle.load(open("data_train_data.pkl", "rb"))
                logger.info('loaded data_train_data.pkl')
                return
            data = pickle.load(open('data_process_datacopy.pkl', 'rb'))
            logger.info('loaded data_process_datacopy.pkl')
            print(len(data))
            train_size = int(len(data) / 8 * 7)
            self.data = self.preProcessData(data)
        elif dataName == "val":
            if os.path.exists("data_val_data.pkl"):
                self.data = pickle.load(open("data_val_data.pkl", "rb"))
                self.nl = pickle.load(open("data_val_nl.pkl", "rb"))
                logger.info('loaded data_val_data.pkl and data_val_nl.pkl')
                return
            self.data = self.preProcessData(
                open(self.val_path, "r", encoding='utf-8'))
        else:
            return

    def Load_Voc(self):
        if os.path.exists("data_nl_voc.pkl"):
            logger.info('loading data_nl_voc.pkl')
            self.NL_VOCAB = pickle.load(open("data_nl_voc.pkl", "rb"))
        if os.path.exists("data_code_voc.pkl"):
            logger.info('loading data_code_voc.pkl')
            self.CODE_VOCAB = pickle.load(open("data_code_voc.pkl", "rb"))
        if os.path.exists("data_char_voc.pkl"):
            logger.info('loading data_char_voc.pkl')
            self.CHAR_VOCAB = pickle.load(open("data_char_voc.pkl", "rb"))
        self.NL_VOCAB["<emptynode>"] = len(self.NL_VOCAB)
        self.CODE_VOCAB["<emptynode>"] = len(self.CODE_VOCAB)

    def init_dic(self):
        print("initVoc")
        #f = open(self.train_path, "r", encoding='utf-8')
        #lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        nls = []
        rules = []
        data = pickle.load(open('data_process_datacopy.pkl', 'rb'))
        for x in data:
            if len(x['rule']) > self.Code_Len:
                continue
            nls.append(x['input'])
        '''for i in tqdm(range(int(len(lines) / 5))):
            data = lines[5 * i].strip().lower().split()
            nls.append(data)
            rulelist = lines[5 * i + 1].strip().split()
            tmp = []
            for x in rulelist:
                if int(x) >= 10000:
                    tmp.append(data[int(x) - 10000])
            rules.append(tmp)
        f.close()
        nl_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=0)
        code_voc = VocabEntry.from_corpus(rules, size=50000, freq_cutoff=10)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id'''
        code_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=10)
        self.CODE_VOCAB = code_voc.word2id
        for x in self.rule_dict:
            print(x)
            lst = x.strip().lower().split()
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.CODE_VOCAB:
                    self.CODE_VOCAB[y] = len(self.CODE_VOCAB)
            #rules.append([lst[0]] + lst[2:])
        # print(self.Code_Voc)
        self.NL_VOCAB = self.CODE_VOCAB
        # print(self.Code_Voc)
        assert ("root" in self.CODE_VOCAB)
        for x in self.NL_VOCAB:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.CHAR_VOCAB:
                    self.CHAR_VOCAB[c] = len(self.CHAR_VOCAB)
        for x in self.CODE_VOCAB:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.CHAR_VOCAB:
                    self.CHAR_VOCAB[c] = len(self.CHAR_VOCAB)
        open("data_nl_voc.pkl", "wb").write(pickle.dumps(self.NL_VOCAB))
        open("data_code_voc.pkl", "wb").write(pickle.dumps(self.CODE_VOCAB))
        open("data_char_voc.pkl", "wb").write(pickle.dumps(self.CHAR_VOCAB))
        print(maxNlLen, maxCodeLen, maxCharLen)

    def get_embedding(self, word_list: List[str], vocab: Dict[str, int]):
        embedding = []
        for word in word_list:
            word = word.lower()
            if word not in vocab:
                embedding.append(1)
            else:
                embedding.append(vocab[word])
        return embedding

    def get_char_embedding(self, word_list: List[str]) -> List[List[int]]:
        embedding = []
        for word in word_list:
            word = word.lower()
            word_emb = []
            for ch in word:
                ch_emb = self.CHAR_VOCAB[ch] if ch in self.CHAR_VOCAB else 1
                word_emb.append(ch_emb)
            embedding.append(word_emb)
        return embedding

    def pad_seq(self, sequence: List[int], max_len: int) -> List[int]:
        '''
        Truncate sequence to max_len,
        if size is smaller than max_len, fill with self.PAD_token

        NOTE mutates sequence
        '''

        if len(sequence) < max_len:
            sequence = sequence + [self.PAD_token] * max_len
            sequence = sequence[:max_len]
        else:
            sequence = sequence[:max_len]

        return sequence

    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_list(self, sequence: list, maxlen1, maxlen2):
        if len(sequence) < maxlen1:
            sequence = sequence + [[self.PAD_token] * maxlen2] * maxlen1
            sequence = sequence[:maxlen1]
        else:
            sequence = sequence[:maxlen1]
        return sequence

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def preProcessOne(self, data_buggy_locations):
        '''
        preprocess data:
        Remove terminal node and their position
        Get embedding, character embedding, 
        '''

        logger.info('starting to pre-process data about buggy location')

        inputNl = []
        inputNlchar = []
        inputPos = []
        inputNlad = []
        Nl: List[List[str]] = []

        for data_buggy_location in data_buggy_locations:

            # 2: treeroot, 1: subroot, 3: prev, 4: after
            node_possibilities: List[int] = data_buggy_location['prob']
            tree_as_str_with_var: str = data_buggy_location['tree']

            node_possibilities = self.pad_seq(node_possibilities, self.Nl_Len)

            # result of tree_as_str_with_var.split():
            # ['MethodDeclaration', 'modifiers', 'public_ter', '^', '^', 'return_type', 'ReferenceType', ...]
            tokens_of_tree_as_str_with_var = tree_as_str_with_var.split()
            Nl.append(tokens_of_tree_as_str_with_var)

            node_root = SimpleNode('root', 0)
            tokens_without_jumps: List[str] = ['root']    # nl without terminal
            nodes_without_jumps: List[SimpleNode] = [node_root]      # nodes without terminal

            # traverse the tree and do sth
            current_node = node_root
            node_id = 1
            for token_as_str in tokens_of_tree_as_str_with_var[1:]:

                if token_as_str != "^":
                    token_as_node = SimpleNode(token_as_str, node_id)
                    node_id += 1

                    token_as_node.father = current_node
                    current_node.children.append(token_as_node)
                    current_node = token_as_node

                    tokens_without_jumps.append(token_as_str)
                    nodes_without_jumps.append(token_as_node)

                else:
                    current_node = current_node.father

            nladrow = []
            nladcol = []
            nladdata = []

            for node_ in nodes_without_jumps:
                if node_.father:

                    if node_.id < self.Nl_Len and node_.father.id < self.Nl_Len:
                        nladrow.append(node_.id)
                        nladcol.append(node_.father.id)
                        nladdata.append(1)

                    for sibling in node_.father.children:
                        if node_.id < self.Nl_Len and sibling.id < self.Nl_Len:
                            nladrow.append(node_.id)
                            nladcol.append(sibling.id)
                            nladdata.append(1)

                for child in node_.children:
                    if node_.id < self.Nl_Len and child.id < self.Nl_Len:
                        nladrow.append(node_.id)
                        nladcol.append(child.id)
                        nladdata.append(1)

            tokens_of_tree_as_str_with_var = tokens_without_jumps

            embedding = self.get_embedding(tokens_of_tree_as_str_with_var, self.NL_VOCAB)
            input_nls = self.pad_seq(embedding, self.Nl_Len)

            nl_ad = sparse.coo_matrix((nladdata, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            input_nl_char = self.get_char_embedding(tokens_of_tree_as_str_with_var)

            for j in range(len(input_nl_char)):
                input_nl_char[j] = self.pad_seq(input_nl_char[j], self.Char_Len)

            input_nl_char = self.pad_list(input_nl_char, self.Nl_Len, self.Char_Len)

            inputNl.append(input_nls)
            inputNlad.append(nl_ad)
            inputPos.append(node_possibilities)
            inputNlchar.append(input_nl_char)

        self.data = [inputNl, inputNlad, inputPos, inputNlchar]
        self.nl = Nl

    def preProcessData(self, dataFile):

        logger.info('starting data pre-processing')

        #lines = dataFile.readlines()
        inputNl = []
        inputNlad = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        inputPos = []
        nls = []
        for i in tqdm(range(len(dataFile))):
            if len(dataFile[i]['rule']) > self.Code_Len:
                continue
            child = {}
            nl = dataFile[i]['input']  # lines[5 * i].lower().strip().split()
            node = SimpleNode('root', 0)
            currnode = node
            idx = 1
            nltmp = ['root']
            nodes = [node]
            for x in nl[1:]:
                if x != "^":
                    nnode = SimpleNode(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.children.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(x.father.id)
                        nladdata.append(1)
                    for s in x.father.children:
                        if x.id < self.Nl_Len and s.id < self.Nl_Len:
                            nladrow.append(x.id)
                            nladcol.append(s.id)
                            nladdata.append(1)
                for s in x.children:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)
            nl = nltmp
            nls.append(dataFile[i]['input'])
            inputpos = dataFile[i]['problist']
            # for j in range(len(inputpos)):
            #    inputpos[j] = inputpos[j]
            inputPos.append(self.pad_seq(inputpos, self.Nl_Len))
            # lines[5 * i + 2].strip().split()
            inputparent = dataFile[i]['fatherlist']
            inputres = dataFile[i]['rule']  # lines[5 * i + 1].strip().split()
            #depth = lines[5 * i + 3].strip().split()
            # lines[5 * i + 4].strip().lower().split()
            parentname = dataFile[i]['fathername']
            for j in range(len(parentname)):
                parentname[j] = parentname[j].lower()
            inputadrow = []
            inputadcol = []
            inputaddata = []
            #inputad = np.zeros([self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])
            inputrule = [self.rule_dict["start -> root"]]
            for j in range(len(inputres)):
                inputres[j] = int(inputres[j])
                inputparent[j] = int(inputparent[j]) + 1
                child.setdefault(inputparent[j], []).append(j + 1)
                if inputres[j] >= 2000000:
                    # assert(0)
                    inputres[j] = len(self.rule_dict) + inputres[j] - 2000000
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.rule_dict))
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.rule_dict['start -> copyword'])
                elif inputres[j] >= 1000000:
                    inputres[j] = len(self.rule_dict) + \
                        inputres[j] - 1000000 + self.Nl_Len
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + j + 1)
                        inputadcol.append(
                            inputres[j] - len(self.rule_dict) - self.Nl_Len)
                        inputaddata.append(1)
                        #inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.rule_dict['start -> copyword2'])
                else:
                    inputrule.append(inputres[j])
                if inputres[j] - len(self.rule_dict) >= self.Nl_Len:
                    print(inputres[j] - len(self.rule_dict))
                if j + 1 < self.Code_Len:
                    inputadrow.append(self.Nl_Len + j + 1)
                    inputadcol.append(self.Nl_Len + inputparent[j])
                    inputaddata.append(1)
                    #inputad[self.Nl_Len + j + 1, self.Nl_Len + inputparent[j]] = 1
            #inputrule = [self.ruledict["start -> Module"]] + inputres
            #depth = self.pad_seq([1] + depth, self.Code_Len)
            inputnls = self.get_embedding(nl, self.NL_VOCAB)
            inputNl.append(self.pad_seq(inputnls, self.Nl_Len))
            inputnlchar = self.get_char_embedding(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(
                inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            inputruleparent = self.pad_seq(self.get_embedding(
                ["start"] + parentname, self.CODE_VOCAB), self.Code_Len)
            inputrulechild = []
            for x in inputrule:
                if x >= len(self.rule_reverse_dict):
                    inputrulechild.append(self.pad_seq(self.get_embedding(
                        ["copyword"], self.CODE_VOCAB), self.Char_Len))
                else:
                    rule = self.rule_reverse_dict[x].strip().lower().split()
                    inputrulechild.append(self.pad_seq(
                        self.get_embedding(rule[2:], self.CODE_VOCAB), self.Char_Len))

            inputparentpath = []
            for j in range(len(inputres)):
                if inputres[j] in self.rule_reverse_dict:
                    tmppath = [self.rule_reverse_dict[inputres[j]].strip().lower().split()[
                        0]]
                    if tmppath[0] != parentname[j].lower() and tmppath[0] == 'statements' and parentname[j].lower() == 'root':
                        # print(tmppath, parentname[j].lower())
                        tmppath[0] = 'root'
                    if tmppath[0] != parentname[j].lower() and tmppath[0] == 'start':
                        tmppath[0] = parentname[j].lower()
                    #print(tmppath, parentname[j].lower(), inputres)
                    assert (tmppath[0] == parentname[j].lower())
                else:
                    tmppath = [parentname[j].lower()]
                '''siblings = child[inputparent[j]]
                for x in siblings:
                    if x == j + 1:
                        break
                    tmppath.append(parentname[x - 1])'''
                # print(inputparent[j])
                curr = inputparent[j]
                while curr != 0:
                    if inputres[curr - 1] >= len(self.rule_reverse_dict):
                        #print(parentname[curr - 1].lower())
                        rule = 'root'
                        # assert(0)
                    else:
                        rule = self.rule_reverse_dict[inputres[curr - 1]
                                                      ].strip().lower().split()[0]
                    # print(rule)
                    tmppath.append(rule)
                    curr = inputparent[curr - 1]
                # print(tmppath)
                inputparentpath.append(self.pad_seq(
                    self.get_embedding(tmppath, self.CODE_VOCAB), 10))
            # assert(0)
            inputrule = self.pad_seq(inputrule, self.Code_Len)
            inputres = self.pad_seq(inputres, self.Code_Len)
            tmp = [self.pad_seq(self.get_embedding(
                ['start'], self.CODE_VOCAB), 10)] + inputparentpath
            inputrulechild = self.pad_list(tmp, self.Code_Len, 10)
            inputRuleParent.append(inputruleparent)
            inputRuleChild.append(inputrulechild)
            inputRes.append(inputres)
            inputRule.append(inputrule)
            inputparent = [0] + inputparent
            inputad = sparse.coo_matrix((inputaddata, (inputadrow, inputadcol)), shape=(
                self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len))
            inputParent.append(inputad)
            inputParentPath.append(self.pad_list(
                inputparentpath, self.Code_Len, 10))
            nlad = sparse.coo_matrix(
                (nladdata, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            inputNlad.append(nlad)
        batchs = [inputNl, inputNlad, inputRule, inputRuleParent, inputRuleChild,
                  inputRes, inputParent, inputParentPath, inputPos, inputNlChar]
        self.data = batchs
        self.nl = nls
        #self.code = codes
        if self.dataName == "train":
            open("data_train_data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("data_train_nl.pkl", "wb").write(pickle.dumps(nls))
            logger.info('saved data_train_data.pkl and data_train_nl.pkl')
        if self.dataName == "val":
            open("data_val_data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("data_val_nl.pkl", "wb").write(pickle.dumps(nls))
            logger.info('saved data_val_data.pkl and data_val_nl.pkl')
        if self.dataName == "test":
            open("data_test_data.pkl", "wb").write(pickle.dumps(batchs))
            #open("data_testcode.pkl", "wb").write(pickle.dumps(self.code))
            open("data_test_nl.pkl", "wb").write(pickle.dumps(self.nl))
            logger.info('saved data_test_data.pkl and data_test_nl.pkl')
        return batchs

    def __getitem__(self, offset):
        ans = []
        '''if self.dataName == "train":
            h5f = h5py.File("data.h5", 'r')
        if self.dataName == "val":
            h5f = h5py.File("valdata.h5", 'r')
        if self.dataName == "test":
            h5f = h5py.File("testdata.h5", 'r')'''
        for i in range(len(self.data)):
            d = self.data[i][offset]
            if i == 1 or i == 6:
                tmp = d.toarray().astype(np.int32)
                ans.append(tmp)
            else:
                ans.append(np.array(d))
            '''if i == 6:
                #print(self.data[i][offset])
                tmp = np.eye(self.Code_Len)[d]
                #print(tmp.shape)
                tmp = np.concatenate([tmp, np.zeros([self.Code_Len, self.Code_Len])], axis=0)[:self.Code_Len,:]#self.pad_list(tmp, self.Code_Len, self.Code_Len)
                ans.append(np.array(tmp))
            else:'''
        return ans

    def __len__(self):
        return len(self.data[0])


class SimpleNode:
    def __init__(self, name: str, id_: int):
        self.name = name
        self.id = id_
        self.father: SimpleNode = None
        self.children: List[SimpleNode] = []
        self.sibiling = None

#dset = SumDataset(args)
