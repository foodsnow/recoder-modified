import pickle
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from Searchnode1 import Node
from base_logger import logger

ONE_LIST = [
    'SRoot',
    'arguments',
    'parameters',
    'body',
    'block',
    'selectors',
    'cases',
    'statements',
    'throws',
    'initializers',
    'declarators',
    'annotations',
    'prefix_operators',
    'postfix_operators',
    'catches',
    'types',
    'dimensions',
    'modifiers',
    'case',
    'finally_block',
    'type_parameters'
]

RULE_LIST = []
FATHER_LIST = []
FATHER_NAMES = []
DEPTH_LIST = []
COPY_NODE = {}
RULES = pickle.load(open("rule.pkl", "rb"))

assert ('value -> <string>_ter' in RULES)

CNUM = len(RULES)
RULEAD = np.zeros([CNUM, CNUM])

LINE_NODE = [
    'Statement_ter',
    'BreakStatement_ter',
    'ReturnStatement_ter',
    'ContinueStatement',
    'ContinueStatement_ter',
    'LocalVariableDeclaration',
    'condition',
    'control',
    'BreakStatement',
    'ContinueStatement',
    'ReturnStatement',
    "parameters",
    'StatementExpression',
    'return_type'
]

REVERSE_RULES_DICT = {}
for x in RULES:
    REVERSE_RULES_DICT[RULES[x]] = x

HAS_COPY: Dict = {}
IS_VALID = True
ACTION = []
RES_LIST = []
N = 0


def find_all(sub_string: str, super_string: str) -> List[int]:
    index_list = []
    index = super_string.find(sub_string)

    while index != -1:
        index_list.append(index)
        index = super_string.find(sub_string, index+1)

    if len(index_list) > 0:
        return index_list
    else:
        return []


def get_copy_id(tokens: List[str], name: str, idx: int):
    original = " ".join(tokens)
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


def get_local_var_names(node: Node) -> Tuple[str, Node]:
    '''
    NOTE recursive: down to children
    '''

    var_names = []

    if node.name == 'VariableDeclarator':
        currnode = -1
        for child in node.child:
            if child.name == 'name':
                currnode = child
                break
        var_names.append((currnode.child[0].name, node))

    if node.name == 'FormalParameter':
        currnode = -1
        for child in node.child:
            if child.name == 'name':
                currnode = child
                break
        var_names.append((currnode.child[0].name, node))

    if node.name == 'InferredFormalParameter':
        currnode = -1
        for child in node.child:
            if child.name == 'name':
                currnode = child
                break
        var_names.append((currnode.child[0].name, node))

    # recursive call
    for child in node.child:
        var_names.extend(get_local_var_names(child))

    return var_names


def getRule(
        node: Node,
        tokens: List[str],
        current_id: int,
        d: int,
        idx: int,
        var_dict: Dict[str, str],
        copy=True,
        calvalid=True) -> Union[Tuple, None]:

    logger.info('starting get_rule()')

    global RULES
    global ONE_LIST
    global RULE_LIST
    global FATHER_LIST
    global DEPTH_LIST
    global COPY_NODE
    global RULEAD
    global IS_VALID

    if not IS_VALID:
        return

    if len(node.child) == 0:
        return [], []

    copyid = -1
    children = node.child

    # case 1
    if len(node.child) == 1 and len(node.child[0].child) == 0 and copyid == -1 and copy:
        if node.child[0].name in var_dict:
            rule = node.name + " -> " + var_dict[node.child[0].name]

            if rule in RULES:
                RULE_LIST.append(RULES[rule])
            else:
                IS_VALID = False
                return

            FATHER_LIST.append(current_id)
            FATHER_NAMES.append(node.name)
            DEPTH_LIST.append(d)
            return

    # case 2
    if copyid == -1:
        copyid = get_copy_id(tokens, node.getTreestr(), node.id)

        if node.name == 'MemberReference' or node.name == 'operator' or \
                node.name == 'type' or node.name == 'prefix_operators' or \
                node.name == 'value':
            copyid = -1

        if node.name == 'operandl' or node.name == 'operandr':
            if node.child[0].name == 'MemberReference' and \
                    node.child[0].child[0].name == 'member':
                copyid = -1

        if node.name == 'Literal':
            if 'value -> ' + node.child[0].child[0].name in RULES:
                copyid = -1

    if len(node.child) == 1 and len(node.child[0].child) == 0 and copyid == -1:
        rule = node.name + " -> " + node.child[0].name
        if rule not in RULES and (node.name == 'member' or node.name == 'qualifier'):
            rule = RULES['start -> unknown']
            RULE_LIST.append(rule)
            FATHER_LIST.append(current_id)
            FATHER_NAMES.append(node.name)
            DEPTH_LIST.append(d)
            return

    # case 3
    if copyid != -1:
        COPY_NODE[node.name] = 1
        RULE_LIST.append(copyid)
        FATHER_LIST.append(current_id)
        FATHER_NAMES.append(node.name)
        DEPTH_LIST.append(d)
        currid = len(RULE_LIST) - 1
        if RULE_LIST[current_id] >= CNUM:
            pass
        elif current_id != -1:
            RULEAD[RULE_LIST[current_id], RULES['start -> copyword']] = 1
            RULEAD[RULES['start -> copyword'], RULE_LIST[current_id]] = 1
        else:
            RULEAD[RULES['start -> copyword'], RULES['start -> root']] = 1
            RULEAD[RULES['start -> root'], RULES['start -> copyword']] = 1
        return
    else:
        if node.name not in ONE_LIST:
            rule = node.name + " -> "
            for x in children:
                rule += x.name + " "
            rule = rule.strip()
            if rule in RULES:
                RULE_LIST.append(RULES[rule])
            else:
                IS_VALID = False
                return
            FATHER_LIST.append(current_id)
            FATHER_NAMES.append(node.name)
            DEPTH_LIST.append(d)
            if RULE_LIST[-1] < CNUM and RULE_LIST[current_id] < CNUM:
                if current_id != -1:
                    RULEAD[RULE_LIST[current_id], RULE_LIST[-1]] = 1
                    RULEAD[RULE_LIST[-1], RULE_LIST[current_id]] = 1
                else:
                    RULEAD[RULES['start -> root'], RULE_LIST[-1]] = 1
                    RULEAD[RULE_LIST[-1], RULES['start -> root']] = 1
            currid = len(RULE_LIST) - 1
            for x in children:
                getRule(x, tokens, currid, d + 1, idx, var_dict)
        else:
            for x in children:
                rule = node.name + " -> " + x.name
                rule = rule.strip()
                if rule in RULES:
                    RULE_LIST.append(RULES[rule])
                else:
                    IS_VALID = False
                    return
                if RULE_LIST[-1] < CNUM and RULE_LIST[current_id] < CNUM:
                    RULEAD[RULE_LIST[current_id], RULE_LIST[-1]] = 1
                    RULEAD[RULE_LIST[-1], RULE_LIST[current_id]] = 1
                FATHER_LIST.append(current_id)
                FATHER_NAMES.append(node.name)
                DEPTH_LIST.append(d)
                getRule(x, tokens, len(RULE_LIST) - 1, d + 1, idx, var_dict)
            rule = node.name + " -> End "
            rule = rule.strip()
            if rule in RULES:
                RULE_LIST.append(RULES[rule])
            else:
                assert (0)
                RULES[rule] = len(RULES)
                RULE_LIST.append(RULES[rule])
            RULEAD[RULE_LIST[current_id], RULE_LIST[-1]] = 1
            RULEAD[RULE_LIST[-1], RULE_LIST[current_id]] = 1
            FATHER_LIST.append(current_id)
            FATHER_NAMES.append(node.name)
            DEPTH_LIST.append(d)


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


def setProb(r, p):
    r.possibility = p
    for x in r.child:
        setProb(x, p)


def get_line_nodes(root_node: Node, block: str) -> List[Node]:
    '''
    return all the nodes (with their children) in root_node
    that have names one of in LINE_NODE

    recursive
    modifies tree: sets .block attribute
    '''

    global LINE_NODE

    line_nodes = []
    block = block + root_node.name
    for child in root_node.child:
        if child.name in LINE_NODE:
            if 'info' in child.getTreestr() or 'assert' in child.getTreestr() or \
                    'logger' in child.getTreestr() or 'LOGGER' in child.getTreestr() or 'system.out' in child.getTreestr().lower():
                continue
            child.block = block
            line_nodes.append(child)
        else:
            tmp = get_line_nodes(child, block)
            line_nodes.extend(tmp)
    return line_nodes


def setid(root):
    global N
    root.id = N
    N += 1
    for x in root.child:
        setid(x)


def isexpanded(lst):
    ans = False
    for x in lst:
        ans = ans or x.expanded
    return ans


def is_changed(node1: Node, node2: Node) -> bool:

    if node1.name != node2.name:
        return False

    if node1 == node2:
        return True

    if node1.name == 'MemberReference' or node1.name == 'BasicType' or \
            node1.name == 'operator' or node1.name == 'qualifier' or \
            node1.name == 'member' or node1.name == 'Literal':
        return True

    if len(node1.child) != len(node2.child):
        return False

    ans = True
    for i in range(len(node1.child)):
        child1 = node1.child[i]
        child2 = node2.child[i]
        ans = ans and is_changed(child1, child2)

    return ans


def get_changed_nodes(node1: Node, node2: Node) -> List[Tuple[Node, Node]]:

    if node1 == node2:
        return []

    ans = []
    if node1.name == 'MemberReference' or node1.name == 'BasicType' or \
            node1.name == 'operator' or node1.name == 'qualifier' or \
            node1.name == 'member' or node1.name == 'Literal':
        return [(node1, node2)]

    for i in range(len(node1.child)):
        ans.extend(get_changed_nodes(node1.child[i], node2.child[i]))

    return ans


def getDiffNode(
        line_nodes_old_tree: List[Node],
        line_nodes_new_tree: List[Node],
        root_node_old_tree: Node,
        old_tree_tokens: List[str],
        method_name: str):

    logger.info('starting get_diff_node()')

    global RES_LIST
    global RULES
    global ONE_LIST
    global RULE_LIST
    global FATHER_LIST
    global DEPTH_LIST
    global COPY_NODE
    global RULEAD
    global FATHER_NAMES
    global N
    global IS_VALID

    # step 1
    delete_node = []
    dic = {}
    dic2 = {}
    for i, lnode_old in enumerate(line_nodes_old_tree):

        has_same = False
        for j, lnode_new in enumerate(line_nodes_new_tree):

            if lnode_old == lnode_new and not lnode_new.expanded and not has_same:
                lnode_new.expanded = True
                lnode_old.expanded = True
                dic[i] = j
                dic2[j] = i
                has_same = True
                continue

            if lnode_old == lnode_new and not lnode_new.expanded and has_same:
                if i - 1 in dic and dic[i - 1] == j - 1:
                    has_same = True
                    line_nodes_new_tree[dic[i]].expanded = False
                    lnode_new.expanded = True
                    del dic2[dic[i]]
                    dic[i] = j
                    dic2[j] = i
                    break

        if not has_same:
            delete_node.append(lnode_old)

    if len(delete_node) > 1:
        return

    # step 2
    pre_id_dict = {}
    after_id_dict = {}
    pre_id = -1
    for i in range(len(line_nodes_old_tree)):
        if line_nodes_old_tree[i].expanded:
            pre_id = i
        else:
            pre_id_dict[i] = pre_id

    after_id = len(line_nodes_old_tree)
    dic[after_id] = len(line_nodes_new_tree)
    dic[-1] = -1
    for i in range(len(line_nodes_old_tree) - 1, -1, -1):
        if line_nodes_old_tree[i].expanded:
            after_id = i
        else:
            after_id_dict[i] = after_id

    for i in range(len(line_nodes_old_tree)):
        if line_nodes_old_tree[i].expanded:
            continue
        else:
            pre_id = pre_id_dict[i]
            after_id = after_id_dict[i]
            pre_id2 = dic[pre_id_dict[i]]
            after_id2 = dic[after_id_dict[i]]

            if pre_id + 2 == after_id and pre_id2 + 2 == after_id2:
                troot = root_node_old_tree

                # this part of the code is similar to testDefect4j.py
                # num_tokens(troot) >= 1000
                if len(root_node_old_tree.getTreestr().strip().split()) >= 1000:
                    temp_lnode_old = line_nodes_old_tree[pre_id + 1]

                    if len(temp_lnode_old.getTreestr().split()) >= 1000:
                        continue

                    last_temp_lnode_old = None
                    while True:
                        if len(temp_lnode_old.getTreestr().split()) >= 1000:
                            break
                        last_temp_lnode_old = temp_lnode_old
                        temp_lnode_old = temp_lnode_old.father

                    ans_root_node = Node(temp_lnode_old.name, 0)
                    ans_root_node.child.append(last_temp_lnode_old)
                    ans_root_node.num = 2 + len(last_temp_lnode_old.getTreestr().strip().split())

                    while True:
                        some_flag = True
                        after_node_idx = temp_lnode_old.child.index(ans_root_node.child[-1]) + 1
                        if after_node_idx < len(temp_lnode_old.child) and ans_root_node.num + temp_lnode_old.child[after_node_idx].getNum() < 1000:
                            some_flag = False
                            ans_root_node.child.append(temp_lnode_old.child[after_node_idx])
                            ans_root_node.num += temp_lnode_old.child[after_node_idx].getNum()
                        prenode = temp_lnode_old.child.index(ans_root_node.child[0]) - 1
                        if prenode >= 0 and ans_root_node.num + temp_lnode_old.child[prenode].getNum() < 1000:
                            some_flag = False
                            ans_root_node.child = [temp_lnode_old.child[prenode]] + ans_root_node.child
                            ans_root_node.num += temp_lnode_old.child[prenode].getNum()
                        if some_flag:
                            break
                    troot = ans_root_node

                for k in range(pre_id + 1, after_id):
                    line_nodes_old_tree[k].expanded = True
                    setProb(line_nodes_old_tree[k], 1)
                if pre_id >= 0:
                    setProb(line_nodes_old_tree[pre_id], 3)
                if after_id < len(line_nodes_old_tree):
                    setProb(line_nodes_old_tree[after_id], 4)

                old_tree_tokens = troot.getTreestr().split()
                N = 0
                setid(troot)

                local_var_names = get_local_var_names(troot)
                fnum = -1
                vnum = -1
                var_dict: Dict[str, str] = {}
                var_dict[method_name] = 'meth0'

                for local_var_name in local_var_names:
                    if local_var_name[1].name == 'VariableDeclarator':
                        vnum += 1
                        var_dict[local_var_name[0]] = 'loc' + str(vnum)
                    else:
                        fnum += 1
                        var_dict[local_var_name[0]] = 'par' + str(fnum)

                RULE_LIST.append(RULES['root -> modified'])
                FATHER_NAMES.append('root')
                FATHER_LIST.append(-1)

                if is_changed(line_nodes_old_tree[pre_id + 1], line_nodes_new_tree[pre_id2 + 1]) and \
                        len(get_changed_nodes(line_nodes_old_tree[pre_id + 1], line_nodes_new_tree[pre_id2 + 1])) <= 1:
                    changed_nodes = get_changed_nodes(line_nodes_old_tree[pre_id + 1], line_nodes_new_tree[pre_id2 + 1])

                    for ch_node in changed_nodes:
                        RULE_LIST.append(1000000 + ch_node[0].id)
                        FATHER_NAMES.append('root')
                        FATHER_LIST.append(-1)

                        if ch_node[0].name == 'BasicType' or ch_node[0].name == 'operator':
                            getRule(
                                node=ch_node[1],
                                tokens=old_tree_tokens,
                                current_id=len(RULE_LIST) - 1,
                                d=0,
                                idx=0,
                                var_dict=var_dict,
                                copy=False,
                                calvalid=False
                            )
                        else:
                            getRule(
                                node=ch_node[1],
                                tokens=old_tree_tokens,
                                current_id=len(RULE_LIST) - 1,
                                d=0,
                                idx=0,
                                var_dict=var_dict,
                                calvalid=False
                            )

                    RULE_LIST.append(RULES['root -> End'])
                    FATHER_LIST.append(-1)
                    FATHER_NAMES.append('root')

                    RES_LIST.append(
                        {
                            'input': root_node_old_tree.printTreeWithVar(troot, var_dict).strip().split(),
                            'rule': RULE_LIST,
                            'problist': root_node_old_tree.getTreeProb(troot),
                            'fatherlist': FATHER_LIST,
                            'fathername': FATHER_NAMES,
                            'vardic': var_dict
                        }
                    )

                    RULE_LIST = []
                    FATHER_NAMES = []
                    FATHER_LIST = []

                    setProb(root_node_old_tree, 2)

                    continue

                for k in range(pre_id2 + 1, after_id2):
                    line_nodes_new_tree[k].expanded = True

                    if line_nodes_new_tree[k].name == 'condition':
                        rule = 'root -> ' + line_nodes_new_tree[k].father.name
                    else:
                        rule = 'root -> ' + line_nodes_new_tree[k].name

                    if rule not in RULES:
                        RULES[rule] = len(RULES)

                    RULE_LIST.append(RULES[rule])
                    FATHER_NAMES.append('root')
                    FATHER_LIST.append(-1)

                    if line_nodes_new_tree[k].name == 'condition':
                        tmpnode = Node(line_nodes_new_tree[k].father.name, 0)
                        tmpnode.child.append(line_nodes_new_tree[k])
                        getRule(
                            node=tmpnode,
                            tokens=old_tree_tokens,
                            current_id=len(RULE_LIST) - 1,
                            d=0,
                            idx=0,
                            var_dict=var_dict
                        )
                    else:
                        getRule(
                            node=line_nodes_new_tree[k],
                            tokens=old_tree_tokens,
                            current_id=len(RULE_LIST) - 1,
                            d=0,
                            idx=0,
                            var_dict=var_dict
                        )

                if not IS_VALID:
                    IS_VALID = True
                    RULE_LIST = []
                    FATHER_NAMES = []
                    FATHER_LIST = []
                    setProb(root_node_old_tree, 2)
                    continue

                RULE_LIST.append(RULES['root -> End'])
                FATHER_LIST.append(-1)
                FATHER_NAMES.append('root')

                assert len(root_node_old_tree.printTree(troot).strip().split()) <= 1000

                RES_LIST.append(
                    {
                        'input': root_node_old_tree.printTreeWithVar(troot, var_dict).strip().split(),
                        'rule': RULE_LIST,
                        'problist': root_node_old_tree.getTreeProb(troot),
                        'fatherlist': FATHER_LIST,
                        'fathername': FATHER_NAMES
                    }
                )

                RULE_LIST = []
                FATHER_NAMES = []
                FATHER_LIST = []
                setProb(root_node_old_tree, 2)
                IS_VALID = True
                continue

            else:
                continue

    # step 3
    pre_id_dict = {}
    after_id_dict = {}
    pre_id = -1
    for i in range(len(line_nodes_new_tree)):
        if line_nodes_new_tree[i].expanded:
            pre_id = i
        else:
            pre_id_dict[i] = pre_id

    after_id = len(line_nodes_new_tree)
    dic2[after_id] = len(line_nodes_old_tree)
    dic2[-1] = -1

    for i in range(len(line_nodes_new_tree) - 1, -1, -1):
        if line_nodes_new_tree[i].expanded:
            after_id = i
        else:
            after_id_dict[i] = after_id

    for i in range(len(line_nodes_new_tree)):
        if line_nodes_new_tree[i].expanded:
            continue
        else:
            pre_id = pre_id_dict[i]
            after_id = after_id_dict[i]

            if pre_id_dict[i] not in dic2:
                return

            pre_id2 = dic2[pre_id_dict[i]]
            if after_id_dict[i] not in dic2:
                return

            after_id2 = dic2[after_id_dict[i]]
            if pre_id2 + 1 != after_id2:
                continue

            troot = root_node_old_tree
            if len(root_node_old_tree.getTreestr().strip().split()) >= 1000:
                if pre_id2 >= 0:
                    temp_lnode2 = line_nodes_old_tree[pre_id2]
                elif after_id2 < len(line_nodes_old_tree):
                    temp_lnode2 = line_nodes_old_tree[after_id2]
                else:
                    assert (0)

                if len(temp_lnode2.getTreestr().split()) >= 1000:
                    continue

                lasttmp = None
                while True:
                    if len(temp_lnode2.getTreestr().split()) >= 1000:
                        break
                    lasttmp = temp_lnode2
                    temp_lnode2 = temp_lnode2.father

                ansroot = Node(temp_lnode2.name, 0)
                ansroot.child.append(lasttmp)
                ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
                while True:
                    b = True
                    after_node_idx = temp_lnode2.child.index(ansroot.child[-1]) + 1
                    if after_node_idx < len(temp_lnode2.child) and ansroot.num + temp_lnode2.child[after_node_idx].getNum() < 1000:
                        b = False
                        ansroot.child.append(temp_lnode2.child[after_node_idx])
                        ansroot.num += temp_lnode2.child[after_node_idx].getNum()
                    prenode = temp_lnode2.child.index(ansroot.child[0]) - 1
                    if prenode >= 0 and ansroot.num + temp_lnode2.child[prenode].getNum() < 1000:
                        b = False
                        ansroot.child = [temp_lnode2.child[prenode]] + ansroot.child
                        ansroot.num += temp_lnode2.child[prenode].getNum()
                    if b:
                        break
                troot = ansroot

            old_tree_tokens = troot.getTreestr().split()
            N = 0
            setid(troot)
            local_var_names = get_local_var_names(troot)
            fnum = -1
            vnum = -1
            var_dict = {}
            var_dict[method_name] = 'meth0'
            for x in local_var_names:
                if x[1].name == 'VariableDeclarator':
                    vnum += 1
                    var_dict[x[0]] = 'loc' + str(vnum)
                else:
                    fnum += 1
                    var_dict[x[0]] = 'par' + str(fnum)
            if pre_id2 >= 0:
                setProb(line_nodes_old_tree[pre_id2], 3)
            if after_id2 < len(line_nodes_old_tree):
                setProb(line_nodes_old_tree[after_id2], 1)
            if after_id2 + 1 < len(line_nodes_old_tree):
                setProb(line_nodes_old_tree[after_id2 + 1], 4)
            RULE_LIST.append(RULES['root -> add'])
            FATHER_NAMES.append('root')
            FATHER_LIST.append(-1)
            for k in range(pre_id + 1, after_id):
                line_nodes_new_tree[k].expanded = True
                if line_nodes_new_tree[k].name == 'condition':
                    rule = 'root -> ' + line_nodes_new_tree[k].father.name
                else:
                    rule = 'root -> ' + line_nodes_new_tree[k].name
                if rule not in RULES:
                    RULES[rule] = len(RULES)
                RULE_LIST.append(RULES[rule])
                FATHER_NAMES.append('root')
                FATHER_LIST.append(-1)
                if line_nodes_new_tree[k].name == 'condition':
                    tmpnode = Node(line_nodes_new_tree[k].father.name, 0)
                    tmpnode.child.append(line_nodes_new_tree[k])
                    getRule(tmpnode, old_tree_tokens, len(RULE_LIST) - 1, 0, 0, var_dict)
                else:
                    getRule(line_nodes_new_tree[k], old_tree_tokens, len(RULE_LIST) - 1, 0, 0, var_dict)
            if not IS_VALID:
                IS_VALID = True
                RULE_LIST = []
                FATHER_NAMES = []
                FATHER_LIST = []
                setProb(root_node_old_tree, 2)
                continue
            RULE_LIST.append(RULES['root -> End'])
            FATHER_LIST.append(-1)
            FATHER_NAMES.append('root')
            assert (len(root_node_old_tree.printTree(troot).strip().split()) <= 1000)
            RES_LIST.append({'input': root_node_old_tree.printTreeWithVar(troot, var_dict).strip().split(), 'rule': RULE_LIST, 'problist': root_node_old_tree.getTreeProb(troot), 'fatherlist': FATHER_LIST, 'fathername': FATHER_NAMES})
            RULE_LIST = []
            FATHER_NAMES = []
            FATHER_LIST = []
            setProb(root_node_old_tree, 2)


lst = ['Chart-1', 'Chart-4', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-10', 'Closure-14', 'Closure-18', 'Closure-20', 'Closure-31', 'Closure-38', 'Closure-51', 'Closure-52', 'Closure-55', 'Closure-57', 'Closure-59', 'Closure-62', 'Closure-71', 'Closure-73', 'Closure-86', 'Closure-104', 'Closure-107', 'Closure-113', 'Closure-123', 'Closure-124', 'Closure-125', 'Closure-130', 'Closure-133', 'Lang-6', 'Lang-16', 'Lang-24', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-3', 'Math-5', 'Math-11', 'Math-27', 'Math-30', 'Math-32', 'Math-33', 'Math-34', 'Math-41', 'Math-48', 'Math-53', 'Math-57', 'Math-58', 'Math-59', 'Math-63', 'Math-69', 'Math-70', 'Math-73', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-96', 'Math-101', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Time-27', 'Lang-43', 'Math-50', 'Math-98', 'Time-7', 'Mockito-38']

if __name__ == '__main__':

    logger.info('starting run solve replace script')

    res = []
    tres = []

    data = []
    data.extend(pickle.load(open('data0_small.pkl', "rb")))

    which_10k = int(sys.argv[1])
    data = data[which_10k * 10000:which_10k*10000 + 10000]

    i = 0
    for data_record_idx, data_record in tqdm(enumerate(data)):
        if 'oldtree' in data_record:
            lines_old = data_record['oldtree']
            lines_new = data_record['newtree']
        else:
            lines_old = data_record['old']
            lines_new = data_record['new']
        i += 1

        lines_old, lines_new = lines_new, lines_old

        # skip if identical pair
        if lines_old.strip().lower() == lines_new.strip().lower():
            continue

        # constructing a tree from old tree tokens
        tokens_old = lines_old.strip().split()
        root_node_old_tree = Node(tokens_old[0], 0)
        current_node = root_node_old_tree
        idx1 = 1
        for j, x in enumerate(tokens_old[1:]):
            if x != "^":
                if tokens_old[j + 2] == '^':
                    x = x + "_ter"
                temp_node = Node(x, idx1)
                idx1 += 1
                temp_node.father = current_node
                current_node.child.append(temp_node)
                current_node = temp_node
            else:
                current_node = current_node.father

        # constructing a tree from new tree tokens
        tokens_new = lines_new.strip().split()
        root_node_new_tree = Node(tokens_new[0], 0)
        current_node = root_node_new_tree
        idx2 = 1
        for j, x in enumerate(tokens_new[1:]):
            if x != "^":
                if tokens_new[j + 2] == '^':
                    x = x + "_ter"
                temp_node = Node(x, idx2)
                idx2 += 1
                temp_node.father = current_node
                current_node.child.append(temp_node)
                current_node = temp_node
            else:
                current_node = current_node.father

        line_nodes_old_tree = get_line_nodes(root_node_old_tree, "")
        line_nodes_new_tree = get_line_nodes(root_node_new_tree, "")

        if len(line_nodes_old_tree) == 0 or len(line_nodes_new_tree) == 0:
            continue

        setProb(root_node_old_tree, 2)

        olen = len(RES_LIST)

        method_name = 'None'
        for x in root_node_old_tree.child:
            if x.name == 'name':
                method_name = x.child[0].name

        getDiffNode(
            line_nodes_old_tree,
            line_nodes_new_tree,
            root_node_old_tree,
            root_node_old_tree.printTree(root_node_old_tree).strip().split(),
            method_name
        )

        if len(RES_LIST) - olen == 1:
            tres.append(RES_LIST[-1])

        if i <= -5:
            assert (0)

        RULE_LIST = []
        FATHER_LIST = []
        FATHER_NAMES = []
        DEPTH_LIST = []
        COPY_NODE = {}
        HAS_COPY = {}
        ACTION = []

    REVERSE_RULES_DICT = {}

    for x in RULES:
        REVERSE_RULES_DICT[RULES[x]] = x

    for p, x in enumerate(tres):
        for rule_idx in x['rule']:
            if rule_idx < 1000000:
                print(REVERSE_RULES_DICT[rule_idx], end=',')
            else:
                if rule_idx >= 2000000:
                    i = rule_idx - 2000000
                else:
                    i = rule_idx - 1000000

    open('rulead%d.pkl' % which_10k, "wb").write(pickle.dumps(RULEAD))
    open('rule%d.pkl' % which_10k, "wb").write(pickle.dumps(RULES))
    open('process_datacopy%d.pkl' % which_10k, "wb").write(pickle.dumps(tres))

    exit(0)
