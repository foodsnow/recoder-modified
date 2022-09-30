import pickle
import math
from Searchnode1 import Node
import numpy as np
from tqdm import tqdm
import pickle
import sys
import re
import random
import time

onelist = ['SRoot', 'arguments', 'parameters', 'body', 'block', 'selectors', 'cases', 'statements', 'throws', 'initializers', 'declarators', 'annotations', 'prefix_operators', 'postfix_operators', 'catches', 'types', 'dimensions', 'modifiers', 'case', 'finally_block', 'type_parameters']
rulelist = []
fatherlist = []
fathername = []
depthlist = []
copynode = {}
rules = pickle.load(open("rule.pkl", "rb"))

assert ('value -> <string>_ter' in rules)

cnum = len(rules)
rulead = np.zeros([cnum, cnum])
linenode = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement', 'ContinueStatement_ter', 'LocalVariableDeclaration', 'condition', 'control', 'BreakStatement', 'ContinueStatement', 'ReturnStatement', "parameters", 'StatementExpression', 'return_type']
rrdict = {}

for x in rules:
    rrdict[rules[x]] = x

hascopy = {}


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index+1)

    if len(index_list) > 0:
        return index_list
    else:
        return []


def getcopyid(nls, name, idx):
    original = " ".join(nls)
    idxs = find_all(name, original)
    if len(idxs) != 0:
        minv = 100000
        idxx = -1
        for x in idxs:
            tmpid = len(original[:x].replace("^", "").split())
            if minv > abs(idx - tmpid):
                minv = abs(idx - tmpid)
                idxx = tmpid
        return 2000000 + idxx
    return -1


def getLocVar(node):
    varnames = []
    if node.name == 'VariableDeclarator':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    if node.name == 'FormalParameter':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    if node.name == 'InferredFormalParameter':
        currnode = -1
        for x in node.child:
            if x.name == 'name':
                currnode = x
                break
        varnames.append((currnode.child[0].name, node))
    for x in node.child:
        varnames.extend(getLocVar(x))
    return varnames


isvalid = True


def getRule(node, nls, currId, d, idx, varnames, copy=True, calvalid=True):
    global rules
    global onelist
    global rulelist
    global fatherlist
    global depthlist
    global copynode
    global rulead
    global isvalid
    if not isvalid:
        return
    if len(node.child) == 0:
        return [], []
    copyid = -1
    child = node.child
    if len(node.child) == 1 and len(node.child[0].child) == 0 and copyid == -1 and copy:
        if node.child[0].name in varnames:
            rule = node.name + " -> " + varnames[node.child[0].name]
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                isvalid = False
                return

            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
            return
    if copyid == -1:
        copyid = getcopyid(nls, node.getTreestr(), node.id)
        if node.name == 'MemberReference' or node.name == 'operator' or node.name == 'type' or node.name == 'prefix_operators' or node.name == 'value':
            copyid = -1
        if node.name == 'operandl' or node.name == 'operandr':
            if node.child[0].name == 'MemberReference' and node.child[0].child[0].name == 'member':
                copyid = -1
        if node.name == 'Literal':
            if 'value -> ' + node.child[0].child[0].name in rules:
                copyid = -1
    if len(node.child) == 1 and len(node.child[0].child) == 0 and copyid == -1:
        rule = node.name + " -> " + node.child[0].name
        if rule not in rules and (node.name == 'member' or node.name == 'qualifier'):
            rule = rules['start -> unknown']
            rulelist.append(rule)
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
            return
    if copyid != -1:
        copynode[node.name] = 1
        rulelist.append(copyid)
        fatherlist.append(currId)
        fathername.append(node.name)
        depthlist.append(d)
        currid = len(rulelist) - 1
        if rulelist[currId] >= cnum:
            pass
        elif currId != -1:
            rulead[rulelist[currId], rules['start -> copyword']] = 1
            rulead[rules['start -> copyword'], rulelist[currId]] = 1
        else:
            rulead[rules['start -> copyword'], rules['start -> root']] = 1
            rulead[rules['start -> root'], rules['start -> copyword']] = 1
        return
    else:
        if node.name not in onelist:
            rule = node.name + " -> "
            for x in child:
                rule += x.name + " "
            rule = rule.strip()
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                print('b', rule)
                isvalid = False
                return
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
            if rulelist[-1] < cnum and rulelist[currId] < cnum:
                if currId != -1:
                    rulead[rulelist[currId], rulelist[-1]] = 1
                    rulead[rulelist[-1], rulelist[currId]] = 1
                else:
                    rulead[rules['start -> root'], rulelist[-1]] = 1
                    rulead[rulelist[-1], rules['start -> root']] = 1
            currid = len(rulelist) - 1
            for x in child:
                getRule(x, nls, currid, d + 1, idx, varnames)
        else:
            for x in (child):
                rule = node.name + " -> " + x.name
                rule = rule.strip()
                if rule in rules:
                    rulelist.append(rules[rule])
                else:
                    print('b', rule)
                    isvalid = False
                    return
                if rulelist[-1] < cnum and rulelist[currId] < cnum:
                    rulead[rulelist[currId], rulelist[-1]] = 1
                    rulead[rulelist[-1], rulelist[currId]] = 1
                fatherlist.append(currId)
                fathername.append(node.name)
                depthlist.append(d)
                getRule(x, nls, len(rulelist) - 1, d + 1, idx, varnames)
            rule = node.name + " -> End "
            rule = rule.strip()
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                print(rule)
                assert (0)
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            rulead[rulelist[currId], rulelist[-1]] = 1
            rulead[rulelist[-1], rulelist[currId]] = 1
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)


def dist(l1, l2):
    if l1[0] != l2[0]:
        return 0
    ans = []
    dic = {}
    for i in range(0, len(l1) + 1):
        dic[(i, 0)] = 0
    for i in range(0, len(l2) + 1):
        dic[(0, i)] = 0
    for i in range(1, len(l1) + 1):
        for j in range(1, len(l2) + 1):
            if l1[i - 1] == l2[j - 1]:
                dic[(i, j)] = dic[(i - 1, j - 1)] + 1
            elif dic[(i - 1, j)] > dic[(i, j - 1)]:
                dic[(i, j)] = dic[(i - 1, j)]
            else:
                dic[(i, j)] = dic[(i, j - 1)]
    return -dic[(len(l1), len(l2))] / min(len(l1), len(l2))


def hassamechild(l1, l2):
    for x in l1.child:
        for y in l2.child:
            if x == y:
                return True
    return False


action = []


def setProb(r, p):
    r.possibility = p
    for x in r.child:
        setProb(x, p)


def getLineNode(root, block, add=True):
    ans = []
    block = block + root.name
    for x in root.child:
        if x.name in linenode:
            if 'info' in x.getTreestr() or 'assert' in x.getTreestr() or 'logger' in x.getTreestr() or 'LOGGER' in x.getTreestr() or 'system.out' in x.getTreestr().lower():
                continue
            x.block = block
            ans.append(x)
        else:
            s = ""
            if not add:
                s = block
            else:
                s = block + root.name
            tmp = getLineNode(x, block)
            ans.extend(tmp)
    return ans


reslist = []
n = 0


def setid(root):
    global n
    root.id = n
    n += 1
    for x in root.child:
        setid(x)


def isexpanded(lst):
    ans = False
    for x in lst:
        ans = ans or x.expanded
    return ans


def ischanged(root1, root2):
    if root1.name != root2.name:
        return False
    if root1 == root2:
        return True
    if root1.name == 'MemberReference' or root1.name == 'BasicType' or root1.name == 'operator' or root1.name == 'qualifier' or root1.name == 'member' or root1.name == 'Literal':
        return True
    if len(root1.child) != len(root2.child):
        return False
    ans = True
    for i in range(len(root1.child)):
        node1 = root1.child[i]
        node2 = root2.child[i]
        ans = ans and ischanged(node1, node2)
    return ans


def getchangednode(root1, root2):
    if root1 == root2:
        return []
    ans = []
    if root1.name == 'MemberReference' or root1.name == 'BasicType' or root1.name == 'operator' or root1.name == 'qualifier' or root1.name == 'member' or root1.name == 'Literal':
        return [(root1, root2)]
    for i in range(len(root1.child)):
        ans.extend(getchangednode(root1.child[i], root2.child[i]))
    return ans


def getDiffNode(linenode1, linenode2, root, nls, m):
    global reslist
    global rules
    global onelist
    global rulelist
    global fatherlist
    global depthlist
    global copynode
    global rulead
    global fathername
    global n
    global isvalid
    deletenode = []
    addnode = []
    node2id = {}
    for i, x in enumerate(linenode1):
        node2id[str(x)] = i
    dic = {}
    dic2 = {}
    for i, x in enumerate(linenode1):
        hasSame = False
        for j, y in enumerate(linenode2):
            if x == y and not y.expanded and not hasSame:
                y.expanded = True
                x.expanded = True
                dic[i] = j
                dic2[j] = i
                hasSame = True
                continue
            if x == y and not y.expanded and hasSame:
                if i - 1 in dic and dic[i - 1] == j - 1:
                    hasSame = True
                    linenode2[dic[i]].expanded = False
                    y.expaned = True
                    del dic2[dic[i]]
                    dic[i] = j
                    dic2[j] = i
                    break
        if not hasSame:
            deletenode.append(x)
    if len(deletenode) > 1:
        return
    preiddict = {}
    afteriddict = {}
    preid = -1
    for i in range(len(linenode1)):
        if linenode1[i].expanded:
            preid = i
        else:
            preiddict[i] = preid
    afterid = len(linenode1)
    dic[afterid] = len(linenode2)
    dic[-1] = -1
    for i in range(len(linenode1) - 1, -1, -1):
        if linenode1[i].expanded:
            afterid = i
        else:
            afteriddict[i] = afterid
    for i in range(len(linenode1)):
        if linenode1[i].expanded:
            continue
        else:
            preid = preiddict[i]
            afterid = afteriddict[i]
            preid2 = dic[preiddict[i]]
            afterid2 = dic[afteriddict[i]]
            if preid + 2 == afterid and preid2 + 2 == afterid2:
                troot = root
                if len(root.getTreestr().strip().split()) >= 1000:
                    tmp = linenode1[preid + 1]
                    if len(tmp.getTreestr().split()) >= 1000:
                        continue
                    lasttmp = None
                    while True:
                        if len(tmp.getTreestr().split()) >= 1000:
                            break
                        lasttmp = tmp
                        tmp = tmp.father
                    index = tmp.child.index(lasttmp)
                    ansroot = Node(tmp.name, 0)
                    ansroot.child.append(lasttmp)
                    ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
                    while True:
                        b = True
                        afternode = tmp.child.index(ansroot.child[-1]) + 1
                        if afternode < len(tmp.child) and ansroot.num + tmp.child[afternode].getNum() < 1000:
                            b = False
                            ansroot.child.append(tmp.child[afternode])
                            ansroot.num += tmp.child[afternode].getNum()
                        prenode = tmp.child.index(ansroot.child[0]) - 1
                        if prenode >= 0 and ansroot.num + tmp.child[prenode].getNum() < 1000:
                            b = False
                            ansroot.child = [tmp.child[prenode]] + ansroot.child
                            ansroot.num += tmp.child[prenode].getNum()
                        if b:
                            break
                    troot = ansroot
                for k in range(preid + 1, afterid):
                    print('--', linenode1[k].printTree(linenode1[k]))
                    linenode1[k].expanded = True
                    setProb(linenode1[k], 1)
                if preid >= 0:
                    setProb(linenode1[preid], 3)
                if afterid < len(linenode1):
                    setProb(linenode1[afterid], 4)
                nls = troot.getTreestr().split()
                n = 0
                setid(troot)
                print('oo', troot.id)
                varnames = getLocVar(troot)
                fnum = -1
                vnum = -1
                vardic = {}
                vardic[m] = 'meth0'
                for x in varnames:
                    if x[1].name == 'VariableDeclarator':
                        vnum += 1
                        vardic[x[0]] = 'loc' + str(vnum)
                    else:
                        fnum += 1
                        vardic[x[0]] = 'par' + str(fnum)
                rulelist.append(rules['root -> modified'])
                fathername.append('root')
                fatherlist.append(-1)
                if ischanged(linenode1[preid + 1], linenode2[preid2 + 1]) and len(getchangednode(linenode1[preid + 1], linenode2[preid2 + 1])) <= 1:
                    nodes = getchangednode(linenode1[preid + 1], linenode2[preid2 + 1])
                    for x in nodes:
                        rulelist.append(1000000 + x[0].id)
                        fathername.append('root')
                        fatherlist.append(-1)
                        if x[0].name == 'BasicType' or x[0].name == 'operator':
                            getRule(x[1], nls, len(rulelist) - 1, 0, 0, vardic, False, calvalid=False)
                        else:
                            getRule(x[1], nls, len(rulelist) - 1, 0, 0, vardic, calvalid=False)
                    rulelist.append(rules['root -> End'])
                    fatherlist.append(-1)
                    fathername.append('root')
                    reslist.append({'input': root.printTreeWithVar(troot, vardic).strip().split(), 'rule': rulelist, 'problist': root.getTreeProb(troot), 'fatherlist': fatherlist, 'fathername': fathername, 'vardic': vardic})
                    rulelist = []
                    fathername = []
                    fatherlist = []
                    setProb(root, 2)
                    continue
                for k in range(preid2 + 1, afterid2):
                    linenode2[k].expanded = True
                    print('--2', linenode2[k].getTreestr())
                    if linenode2[k].name == 'condition':
                        rule = 'root -> ' + linenode2[k].father.name
                    else:
                        rule = 'root -> ' + linenode2[k].name
                    if rule not in rules:
                        rules[rule] = len(rules)
                    rulelist.append(rules[rule])
                    fathername.append('root')
                    fatherlist.append(-1)
                    if linenode2[k].name == 'condition':
                        tmpnode = Node(linenode2[k].father.name, 0)
                        tmpnode.child.append(linenode2[k])
                        getRule(tmpnode, nls, len(rulelist) - 1, 0, 0, vardic)
                    else:
                        getRule(linenode2[k], nls, len(rulelist) - 1, 0, 0, vardic)
                if not isvalid:
                    isvalid = True
                    rulelist = []
                    fathername = []
                    fatherlist = []
                    setProb(root, 2)
                    continue
                rulelist.append(rules['root -> End'])
                fatherlist.append(-1)
                fathername.append('root')
                assert (len(root.printTree(troot).strip().split()) <= 1000)
                reslist.append({'input': root.printTreeWithVar(troot, vardic).strip().split(), 'rule': rulelist, 'problist': root.getTreeProb(troot), 'fatherlist': fatherlist, 'fathername': fathername})
                rulelist = []
                fathername = []
                fatherlist = []
                setProb(root, 2)
                isvalid = True
                continue
            else:
                continue
    preiddict = {}
    afteriddict = {}
    preid = -1
    for i in range(len(linenode2)):
        if linenode2[i].expanded:
            preid = i
        else:
            preiddict[i] = preid
    afterid = len(linenode2)
    dic2[afterid] = len(linenode1)
    dic2[-1] = -1
    print(dic)
    for i in range(len(linenode2) - 1, -1, -1):
        if linenode2[i].expanded:
            afterid = i
        else:
            afteriddict[i] = afterid
    for i in range(len(linenode2)):
        if linenode2[i].expanded:
            continue
        else:
            preid = preiddict[i]
            afterid = afteriddict[i]
            if preiddict[i] not in dic2:
                return
            preid2 = dic2[preiddict[i]]
            if afteriddict[i] not in dic2:
                return
            afterid2 = dic2[afteriddict[i]]
            if preid2 + 1 != afterid2:
                continue
            troot = root
            if len(root.getTreestr().strip().split()) >= 1000:
                if preid2 >= 0:
                    tmp = linenode1[preid2]
                elif afterid2 < len(linenode1):
                    tmp = linenode1[afterid2]
                else:
                    assert (0)
                if len(tmp.getTreestr().split()) >= 1000:
                    continue
                lasttmp = None
                while True:
                    if len(tmp.getTreestr().split()) >= 1000:
                        break
                    lasttmp = tmp
                    tmp = tmp.father
                index = tmp.child.index(lasttmp)
                ansroot = Node(tmp.name, 0)
                ansroot.child.append(lasttmp)
                ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
                while True:
                    b = True
                    afternode = tmp.child.index(ansroot.child[-1]) + 1
                    if afternode < len(tmp.child) and ansroot.num + tmp.child[afternode].getNum() < 1000:
                        b = False
                        ansroot.child.append(tmp.child[afternode])
                        ansroot.num += tmp.child[afternode].getNum()
                    prenode = tmp.child.index(ansroot.child[0]) - 1
                    if prenode >= 0 and ansroot.num + tmp.child[prenode].getNum() < 1000:
                        b = False
                        ansroot.child = [tmp.child[prenode]] + ansroot.child
                        ansroot.num += tmp.child[prenode].getNum()
                    if b:
                        break
                troot = ansroot
            nls = troot.getTreestr().split()
            n = 0
            setid(troot)
            varnames = getLocVar(troot)
            fnum = -1
            vnum = -1
            vardic = {}
            vardic[m] = 'meth0'
            for x in varnames:
                if x[1].name == 'VariableDeclarator':
                    vnum += 1
                    vardic[x[0]] = 'loc' + str(vnum)
                else:
                    fnum += 1
                    vardic[x[0]] = 'par' + str(fnum)
            if preid2 >= 0:
                setProb(linenode1[preid2], 3)
            if afterid2 < len(linenode1):
                setProb(linenode1[afterid2], 1)
            if afterid2 + 1 < len(linenode1):
                setProb(linenode1[afterid2 + 1], 4)
            rulelist.append(rules['root -> add'])
            fathername.append('root')
            fatherlist.append(-1)
            for k in range(preid + 1, afterid):
                linenode2[k].expanded = True
                if linenode2[k].name == 'condition':
                    rule = 'root -> ' + linenode2[k].father.name
                else:
                    rule = 'root -> ' + linenode2[k].name
                if rule not in rules:
                    rules[rule] = len(rules)
                rulelist.append(rules[rule])
                fathername.append('root')
                fatherlist.append(-1)
                if linenode2[k].name == 'condition':
                    tmpnode = Node(linenode2[k].father.name, 0)
                    tmpnode.child.append(linenode2[k])
                    getRule(tmpnode, nls, len(rulelist) - 1, 0, 0, vardic)
                else:
                    getRule(linenode2[k], nls, len(rulelist) - 1, 0, 0, vardic)
            if not isvalid:
                isvalid = True
                rulelist = []
                fathername = []
                fatherlist = []
                setProb(root, 2)
                continue
            rulelist.append(rules['root -> End'])
            fatherlist.append(-1)
            fathername.append('root')
            assert (len(root.printTree(troot).strip().split()) <= 1000)
            reslist.append({'input': root.printTreeWithVar(troot, vardic).strip().split(), 'rule': rulelist, 'problist': root.getTreeProb(troot), 'fatherlist': fatherlist, 'fathername': fathername})
            rulelist = []
            fathername = []
            fatherlist = []
            setProb(root, 2)


lst = ['Chart-1', 'Chart-4', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-10', 'Closure-14', 'Closure-18', 'Closure-20', 'Closure-31', 'Closure-38', 'Closure-51', 'Closure-52', 'Closure-55', 'Closure-57', 'Closure-59', 'Closure-62', 'Closure-71', 'Closure-73', 'Closure-86', 'Closure-104', 'Closure-107', 'Closure-113', 'Closure-123', 'Closure-124', 'Closure-125', 'Closure-130', 'Closure-133', 'Lang-6', 'Lang-16', 'Lang-24', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-3', 'Math-5', 'Math-11', 'Math-27', 'Math-30', 'Math-32', 'Math-33', 'Math-34', 'Math-41', 'Math-48', 'Math-53', 'Math-57', 'Math-58', 'Math-59', 'Math-63', 'Math-69', 'Math-70', 'Math-73', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-96', 'Math-101', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Time-27', 'Lang-43', 'Math-50', 'Math-98', 'Time-7', 'Mockito-38']
if __name__ == '__main__':
    res = []
    tres = []
    data = []
    data.extend(pickle.load(open('data0.pkl', "rb")))
    print(data[0])
    assert (0)
    newdata = []
    v = int(sys.argv[1])
    data = data[v * 10000:v*10000 + 10000]
    i = 0
    for xs in tqdm(data):
        if 'oldtree' in xs:
            lines1 = xs['oldtree']
            lines2 = xs['newtree']
        else:
            lines1 = xs['old']
            lines2 = xs['new']
        i += 1
        lines1, lines2 = lines2, lines1
        if lines1.strip().lower() == lines2.strip().lower():
            continue
        tokens = lines1.strip().split()
        root = Node(tokens[0], 0)
        currnode = root
        idx1 = 1
        for j, x in enumerate(tokens[1:]):
            if x != "^":
                if tokens[j + 2] == '^':
                    x = x + "_ter"
                nnode = Node(x, idx1)
                idx1 += 1
                nnode.father = currnode
                currnode.child.append(nnode)
                currnode = nnode
            else:
                currnode = currnode.father
        root2 = Node(tokens[0], 0)
        currnode = root2
        tokens = lines2.strip().split()
        idx = 1
        for j, x in enumerate(tokens[1:]):
            if x != "^":
                if tokens[j + 2] == '^':
                    x = x + "_ter"
                nnode = Node(x, idx)
                idx += 1
                nnode.father = currnode
                currnode.child.append(nnode)
                currnode = nnode
            else:
                currnode = currnode.father
        linenode1 = getLineNode(root, "")
        linenode2 = getLineNode(root2, "")
        if len(linenode1) == 0 or len(linenode2) == 0:
            continue
        setProb(root, 2)
        olen = len(reslist)
        m = 'None'
        for x in root.child:
            if x.name == 'name':
                m = x.child[0].name
        getDiffNode(linenode1, linenode2, root, root.printTree(root).strip().split(), m)
        if len(reslist) - olen == 1:
            tres.append(reslist[-1])
            newdata.append(xs)
        if i <= -5:
            assert (0)
        rulelist = []
        fatherlist = []
        fathername = []
        depthlist = []
        copynode = {}
        hascopy = {}
        action = []
    rrdict = {}
    for x in rules:
        rrdict[rules[x]] = x
    for p, x in enumerate(tres):
        tmp = []
        for s in x['input']:
            if s != '^':
                tmp.append(s)
        for x in x['rule']:
            if x < 1000000:
                print(rrdict[x], end=',')
            else:
                if x >= 2000000:
                    i = x - 2000000
                else:
                    i = x - 1000000
    open('rulead%d.pkl' % v, "wb").write(pickle.dumps(rulead))
    open('rule2.pkl', "wb").write(pickle.dumps(rules))
    open('process_datacopy%d.pkl' % v, "wb").write(pickle.dumps(tres))
    exit(0)
