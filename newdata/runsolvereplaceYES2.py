import pickle
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from Searchnode1 import Node
from base_logger import logger


ONE_LIST = [
    'SRoot', 'arguments', 'parameters', 'body', 'block', 'selectors', 'cases', 'statements',
    'throws', 'initializers', 'declarators', 'annotations', 'prefix_operators', 'postfix_operators',
    'catches', 'types', 'dimensions', 'modifiers', 'case', 'finally_block', 'type_parameters'
]

LINE_NODES_NAMES = [
    'Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement',
    'ContinueStatement_ter', 'LocalVariableDeclaration', 'condition', 'control',
    'BreakStatement', 'ContinueStatement', 'ReturnStatement', "parameters",
    'StatementExpression', 'return_type'
]

RULE_LIST = []
FATHER_LIST = []
FATHER_NAMES = []
DEPTH_LIST = []
COPY_NODE = {}
HAS_COPY: Dict = {}
IS_VALID = True
ACTION = []
RES_LIST = []
N = 0

RULES: Dict[str, int] = pickle.load(open("rule.pkl", "rb"))
assert ('value -> <string>_ter' in RULES)
NUM_RULES: int = len(RULES)

RULEAD: np.ndarray = np.zeros([NUM_RULES, NUM_RULES])

REVERSE_RULES_DICT: Dict[int, str] = {}
for x in RULES:
    REVERSE_RULES_DICT[RULES[x]] = x


def find_all(sub_string: str, super_string: str) -> List[int]:
    '''
    find all indices of sub_string in super_string
    return as a list
    '''

    index_list = []
    index = super_string.find(sub_string)

    while index != -1:
        index_list.append(index)
        index = super_string.find(sub_string, index+1)

    if len(index_list) > 0:
        return index_list
    else:
        return []


def get_copy_id(tokens: List[str], node_str: str, node_idx: int):
    tokens_str = " ".join(tokens)
    node_str_idxs = find_all(node_str, tokens_str)

    if len(node_str_idxs) != 0:
        minv = 100000
        copy_id = -1

        for node_str_idx in node_str_idxs:
            num_tokens_before = len(tokens_str[:node_str_idx].replace("^", "").split())

            # closest num_tokens_before
            if minv > abs(node_idx - num_tokens_before):
                minv = abs(node_idx - num_tokens_before)
                copy_id = num_tokens_before

        return 2000000 + copy_id

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


def set_prob(r, p):
    r.possibility = p
    for x in r.child:
        set_prob(x, p)


def set_id(root_node: Node):
    '''
    set numerical IDs starting from the root and id=1
    in DFS traversal
    '''

    global N

    root_node.id = N
    N += 1

    for child in root_node.child:
        set_id(child)


def get_line_nodes(root_node: Node, block: str) -> List[Node]:
    '''
    return all the nodes (with their children) in root_node
    that have names one of in LINE_NODE

    recursive
    modifies tree: sets .block attribute
    '''

    global LINE_NODES_NAMES

    line_nodes = []

    block = block + root_node.name

    for child in root_node.child:

        if child.name in LINE_NODES_NAMES:
            # skip certain nodes
            if 'info' in child.getTreestr() or 'assert' in child.getTreestr() or \
                    'logger' in child.getTreestr() or 'LOGGER' in child.getTreestr() or 'system.out' in child.getTreestr().lower():
                continue

            child.block = block
            line_nodes.append(child)
        else:
            lnodes_child = get_line_nodes(child, block)
            line_nodes.extend(lnodes_child)

    return line_nodes


def get_rule(
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
        if RULE_LIST[current_id] >= NUM_RULES:
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
            for _child in node.child:
                rule += _child.name + " "
            rule = rule.strip()

            if rule in RULES:
                RULE_LIST.append(RULES[rule])
            else:
                IS_VALID = False
                return

            FATHER_LIST.append(current_id)
            FATHER_NAMES.append(node.name)
            DEPTH_LIST.append(d)

            if RULE_LIST[-1] < NUM_RULES and RULE_LIST[current_id] < NUM_RULES:
                if current_id != -1:
                    RULEAD[RULE_LIST[current_id], RULE_LIST[-1]] = 1
                    RULEAD[RULE_LIST[-1], RULE_LIST[current_id]] = 1
                else:
                    RULEAD[RULES['start -> root'], RULE_LIST[-1]] = 1
                    RULEAD[RULE_LIST[-1], RULES['start -> root']] = 1

            _rec_current_id = len(RULE_LIST) - 1
            for _child in node.child:
                get_rule(
                    node=_child,
                    tokens=tokens,
                    current_id=_rec_current_id,
                    d=d + 1,
                    idx=idx,
                    var_dict=var_dict
                )

        else:

            for _child in node.child:

                rule = node.name + " -> " + _child.name
                rule = rule.strip()

                if rule in RULES:
                    RULE_LIST.append(RULES[rule])
                else:
                    IS_VALID = False
                    return

                if RULE_LIST[-1] < NUM_RULES and RULE_LIST[current_id] < NUM_RULES:
                    RULEAD[RULE_LIST[current_id], RULE_LIST[-1]] = 1
                    RULEAD[RULE_LIST[-1], RULE_LIST[current_id]] = 1

                FATHER_LIST.append(current_id)
                FATHER_NAMES.append(node.name)
                DEPTH_LIST.append(d)

                get_rule(
                    node=_child,
                    tokens=tokens,
                    current_id=len(RULE_LIST) - 1,
                    d=d + 1,
                    idx=idx,
                    var_dict=var_dict
                )

            rule = node.name + " -> End"
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


def is_changed(node_old: Node, node_new: Node) -> bool:

    # names are not identical -> False
    if node_old.name != node_new.name:
        return False

    # names are identical
    # trees are identical -> True
    if node_old == node_new:
        return True

    # names are identical
    # trees are not identical
    # names are one of the following -> True
    if node_old.name == 'MemberReference' or node_old.name == 'BasicType' or \
            node_old.name == 'operator' or node_old.name == 'qualifier' or \
            node_old.name == 'member' or node_old.name == 'Literal':
        return True

    # names are identical
    # trees are not identical
    # names are not one of the above
    # number of children are not same -> False
    if len(node_old.child) != len(node_new.child):
        return False

    ans = True
    for i in range(len(node_old.child)):
        child1 = node_old.child[i]
        child2 = node_new.child[i]
        ans = ans and is_changed(child1, child2)

    # for the result to be True
    # the above base case conditions should be True for all children
    return ans


def get_changed_nodes(node_old: Node, node_new: Node) -> List[Tuple[Node, Node]]:

    if node_old == node_new:
        return []

    if node_old.name == 'MemberReference' or node_old.name == 'BasicType' or \
            node_old.name == 'operator' or node_old.name == 'qualifier' or \
            node_old.name == 'member' or node_old.name == 'Literal':
        return [(node_old, node_new)]

    changed_nodes = []
    for i in range(len(node_old.child)):
        changed_nodes.extend(get_changed_nodes(node_old.child[i], node_new.child[i]))

    return changed_nodes


def get_diff_node(
        line_nodes_old_tree: List[Node],
        line_nodes_new_tree: List[Node],
        root_node_old_tree: Node,
        troot_tokens_old: List[str],
        method_name: str):

    logger.info('starting get_diff_node()')

    global RES_LIST
    global RULES
    global RULE_LIST
    global FATHER_LIST
    global DEPTH_LIST
    global COPY_NODE
    global RULEAD
    global FATHER_NAMES
    global N
    global IS_VALID

    # step 1
    # do mapping between line nodes in old tree and line nodes in new tree
    unmapped_nodes: List[Node] = []
    map_old2new: Dict[int, int] = {}
    map_new2old: Dict[int, int] = {}

    for i, lnode_old in enumerate(line_nodes_old_tree):

        _has_same = False
        for j, lnode_new in enumerate(line_nodes_new_tree):

            if lnode_old == lnode_new and not lnode_new.mapped and not _has_same:
                lnode_new.mapped = True
                lnode_old.mapped = True
                map_old2new[i] = j
                map_new2old[j] = i
                _has_same = True
                continue

            if lnode_old == lnode_new and not lnode_new.mapped and _has_same:
                if i - 1 in map_old2new and map_old2new[i - 1] == j - 1:
                    _has_same = True
                    line_nodes_new_tree[map_old2new[i]].mapped = False
                    lnode_new.mapped = True
                    del map_new2old[map_old2new[i]]
                    map_old2new[i] = j
                    map_new2old[j] = i
                    break

        if not _has_same:
            unmapped_nodes.append(lnode_old)

    # do not consider cases when there are more than 1 unmapped node
    # from the old tree to the new tree
    if len(unmapped_nodes) > 1:
        return

    # step 2
    # these two dictionaries contain indices to nodes in line_nodes_old_tree
    # before and after the unmapped node
    pre_id_dict_old = {}
    after_id_dict_old = {}

    pre_id = -1
    for i in range(len(line_nodes_old_tree)):
        if line_nodes_old_tree[i].mapped:
            pre_id = i
        else:
            pre_id_dict_old[i] = pre_id

    after_id = len(line_nodes_old_tree)
    map_old2new[after_id] = len(line_nodes_new_tree)
    map_old2new[-1] = -1
    for i in range(len(line_nodes_old_tree) - 1, -1, -1):
        if line_nodes_old_tree[i].mapped:
            after_id = i
        else:
            after_id_dict_old[i] = after_id

    # loop over unmapped nodes in line_nodes_old_tree
    # there might be more than one unmapped nodes
    for i in range(len(line_nodes_old_tree)):
        if line_nodes_old_tree[i].mapped:
            continue

        pre_id_old = pre_id_dict_old[i]
        after_id_old = after_id_dict_old[i]
        pre_id_new = map_old2new[pre_id_dict_old[i]]
        after_id_new = map_old2new[after_id_dict_old[i]]

        # if the unmapped node is in-between pre_id and after_id in both old and new tree
        if pre_id_old + 2 == after_id_old and pre_id_new + 2 == after_id_new:

            troot = root_node_old_tree

            # reconstruct the troot of which unmapped node is a child
            # this part of the code is similar to testDefect4j.py
            # num_tokens(troot) >= 1000
            if len(root_node_old_tree.getTreestr().strip().split()) >= 1000:
                unmapped_lnode_old = line_nodes_old_tree[pre_id_old + 1]

                # skip the iteration if the unmapped line node is too big
                if len(unmapped_lnode_old.getTreestr().split()) >= 1000:
                    continue

                # choose the largest ancestor of unmapped_lnode_old
                # such that its size is less than 1000
                last_temp_lnode_old = None
                while True:
                    if len(unmapped_lnode_old.getTreestr().split()) >= 1000:
                        break
                    last_temp_lnode_old = unmapped_lnode_old
                    unmapped_lnode_old = unmapped_lnode_old.father

                # reconstruct the tree at ans_root_node
                ans_root_node = Node(unmapped_lnode_old.name, 0)
                ans_root_node.child.append(last_temp_lnode_old)
                ans_root_node.num = 2 + len(last_temp_lnode_old.getTreestr().strip().split())

                # add the children until size of 1000 is reached
                # some children may not be added
                while True:
                    some_flag = True

                    after_node_idx = unmapped_lnode_old.child.index(ans_root_node.child[-1]) + 1

                    if after_node_idx < len(unmapped_lnode_old.child) and \
                            ans_root_node.num + unmapped_lnode_old.child[after_node_idx].getNum() < 1000:

                        some_flag = False
                        ans_root_node.child.append(unmapped_lnode_old.child[after_node_idx])
                        ans_root_node.num += unmapped_lnode_old.child[after_node_idx].getNum()

                    pre_node_idx = unmapped_lnode_old.child.index(ans_root_node.child[0]) - 1

                    if pre_node_idx >= 0 and \
                            ans_root_node.num + unmapped_lnode_old.child[pre_node_idx].getNum() < 1000:

                        some_flag = False
                        ans_root_node.child = [unmapped_lnode_old.child[pre_node_idx]] + ans_root_node.child
                        ans_root_node.num += unmapped_lnode_old.child[pre_node_idx].getNum()

                    if some_flag:
                        break

                troot = ans_root_node

            # theoretically, this loop has only one iteration, of k being the index of unmapped node
            for k in range(pre_id_old + 1, after_id_old):
                line_nodes_old_tree[k].mapped = True
                set_prob(line_nodes_old_tree[k], 1)

            if pre_id_old >= 0:
                set_prob(line_nodes_old_tree[pre_id_old], 3)

            if after_id_old < len(line_nodes_old_tree):
                set_prob(line_nodes_old_tree[after_id_old], 4)

            troot_tokens_old = troot.getTreestr().split()
            N = 0
            set_id(troot)

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

            # changed node is one of the following
            # MemberReference, BasicType, operator, qualifier, member, Literal
            if is_changed(line_nodes_old_tree[pre_id_old + 1], line_nodes_new_tree[pre_id_new + 1]) and \
                    len(get_changed_nodes(line_nodes_old_tree[pre_id_old + 1], line_nodes_new_tree[pre_id_new + 1])) <= 1:

                changed_nodes = get_changed_nodes(line_nodes_old_tree[pre_id_old + 1], line_nodes_new_tree[pre_id_new + 1])

                for ch_node in changed_nodes:
                    RULE_LIST.append(1000000 + ch_node[0].id)
                    FATHER_NAMES.append('root')
                    FATHER_LIST.append(-1)

                    if ch_node[0].name == 'BasicType' or ch_node[0].name == 'operator':
                        get_rule(
                            node=ch_node[1],
                            tokens=troot_tokens_old,
                            current_id=len(RULE_LIST) - 1,
                            d=0,
                            idx=0,
                            var_dict=var_dict,
                            copy=False,
                            calvalid=False
                        )
                    else:
                        get_rule(
                            node=ch_node[1],
                            tokens=troot_tokens_old,
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

                set_prob(root_node_old_tree, 2)

                continue

            for k in range(pre_id_new + 1, after_id_new):
                line_nodes_new_tree[k].mapped = True

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
                    get_rule(
                        node=tmpnode,
                        tokens=troot_tokens_old,
                        current_id=len(RULE_LIST) - 1,
                        d=0,
                        idx=0,
                        var_dict=var_dict
                    )
                else:
                    get_rule(
                        node=line_nodes_new_tree[k],
                        tokens=troot_tokens_old,
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
                set_prob(root_node_old_tree, 2)
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
            set_prob(root_node_old_tree, 2)
            IS_VALID = True
            continue

    # step 3
    # these two dictionaries contain indices to nodes in line_nodes_new_tree
    # before and after the unmapped node
    pre_id_dict_new = {}
    after_id_dict_new = {}
    pre_id = -1
    for i in range(len(line_nodes_new_tree)):
        if line_nodes_new_tree[i].mapped:
            pre_id = i
        else:
            pre_id_dict_new[i] = pre_id

    after_id = len(line_nodes_new_tree)
    map_new2old[after_id] = len(line_nodes_old_tree)
    map_new2old[-1] = -1
    for i in range(len(line_nodes_new_tree) - 1, -1, -1):
        if line_nodes_new_tree[i].mapped:
            after_id = i
        else:
            after_id_dict_new[i] = after_id

    # loop over unmapped nodes in line_nodes_old_tree
    # there might be more than one unmapped nodes
    for i in range(len(line_nodes_new_tree)):
        if line_nodes_new_tree[i].mapped:
            continue

        pre_id_new = pre_id_dict_new[i]
        after_id_new = after_id_dict_new[i]

        # if pre node of unmapped node from new tree does not have a mapping pair
        if pre_id_dict_new[i] not in map_new2old:
            return
        pre_id_old = map_new2old[pre_id_dict_new[i]]

        # if after node of unmapped node from new tree does not have a mapping pair
        if after_id_dict_new[i] not in map_new2old:
            return
        after_id_old = map_new2old[after_id_dict_new[i]]

        # skip if pre node and after node do not come one after another
        if pre_id_old + 1 != after_id_old:
            continue

        troot = root_node_old_tree
        if len(root_node_old_tree.getTreestr().strip().split()) >= 1000:

            if pre_id_old >= 0:
                temp_lnode_old = line_nodes_old_tree[pre_id_old]
            elif after_id_old < len(line_nodes_old_tree):
                temp_lnode_old = line_nodes_old_tree[after_id_old]
            else:
                assert (0)

            if len(temp_lnode_old.getTreestr().split()) >= 1000:
                continue

            # choose the largest ancestor of unmapped_lnode_old
            # such that its size is less than 1000
            last_temp_lnode_old2 = None
            while True:
                if len(temp_lnode_old.getTreestr().split()) >= 1000:
                    break
                last_temp_lnode_old2 = temp_lnode_old
                temp_lnode_old = temp_lnode_old.father

            # reconstruct the tree at ans_root_node
            ans_root_node2 = Node(temp_lnode_old.name, 0)
            ans_root_node2.child.append(last_temp_lnode_old2)
            ans_root_node2.num = 2 + len(last_temp_lnode_old2.getTreestr().strip().split())

            # add the children until size of 1000 is reached
            # some children may not be added
            while True:
                some_flag2 = True

                after_node_idx = temp_lnode_old.child.index(ans_root_node2.child[-1]) + 1

                if after_node_idx < len(temp_lnode_old.child) and \
                        ans_root_node2.num + temp_lnode_old.child[after_node_idx].getNum() < 1000:

                    some_flag2 = False
                    ans_root_node2.child.append(temp_lnode_old.child[after_node_idx])
                    ans_root_node2.num += temp_lnode_old.child[after_node_idx].getNum()

                pre_node_idx2 = temp_lnode_old.child.index(ans_root_node2.child[0]) - 1

                if pre_node_idx2 >= 0 and \
                        ans_root_node2.num + temp_lnode_old.child[pre_node_idx2].getNum() < 1000:

                    some_flag2 = False
                    ans_root_node2.child = [temp_lnode_old.child[pre_node_idx2]] + ans_root_node2.child
                    ans_root_node2.num += temp_lnode_old.child[pre_node_idx2].getNum()

                if some_flag2:
                    break

            troot = ans_root_node2

        troot_tokens_old = troot.getTreestr().split()
        N = 0
        set_id(troot)

        local_var_names = get_local_var_names(troot)
        fnum = -1
        vnum = -1
        var_dict = {}
        var_dict[method_name] = 'meth0'

        for local_var_name in local_var_names:
            if local_var_name[1].name == 'VariableDeclarator':
                vnum += 1
                var_dict[local_var_name[0]] = 'loc' + str(vnum)
            else:
                fnum += 1
                var_dict[local_var_name[0]] = 'par' + str(fnum)

        if pre_id_old >= 0:
            set_prob(line_nodes_old_tree[pre_id_old], 3)
        if after_id_old < len(line_nodes_old_tree):
            set_prob(line_nodes_old_tree[after_id_old], 1)
        if after_id_old + 1 < len(line_nodes_old_tree):
            set_prob(line_nodes_old_tree[after_id_old + 1], 4)

        RULE_LIST.append(RULES['root -> add'])
        FATHER_NAMES.append('root')
        FATHER_LIST.append(-1)

        for k in range(pre_id_new + 1, after_id_new):
            line_nodes_new_tree[k].mapped = True

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
                get_rule(
                    node=tmpnode,
                    tokens=troot_tokens_old,
                    current_id=len(RULE_LIST) - 1,
                    d=0,
                    idx=0,
                    var_dict=var_dict
                )
            else:
                get_rule(
                    node=line_nodes_new_tree[k],
                    tokens=troot_tokens_old,
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
            set_prob(root_node_old_tree, 2)
            continue

        RULE_LIST.append(RULES['root -> End'])
        FATHER_LIST.append(-1)
        FATHER_NAMES.append('root')

        assert (len(root_node_old_tree.printTree(troot).strip().split()) <= 1000)

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

        set_prob(root_node_old_tree, 2)


if __name__ == '__main__':

    logger.info('starting run solve replace script')

    tres = []

    data = []
    data.extend(pickle.load(open('data0_small.pkl', "rb")))

    which_10k = int(sys.argv[1])
    data = data[which_10k * 10000:which_10k*10000 + 10000]

    for data_record_idx, data_record in tqdm(enumerate(data)):
        if 'oldtree' in data_record:
            lines_old = data_record['oldtree']
            lines_new = data_record['newtree']
        else:
            lines_old = data_record['old']
            lines_new = data_record['new']

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

        set_prob(root_node_old_tree, 2)

        old_len = len(RES_LIST)

        method_name = 'None'
        for x in root_node_old_tree.child:
            if x.name == 'name':
                method_name = x.child[0].name

        get_diff_node(
            line_nodes_old_tree,
            line_nodes_new_tree,
            root_node_old_tree,
            root_node_old_tree.printTree(root_node_old_tree).strip().split(),
            method_name
        )

        if len(RES_LIST) - old_len == 1:
            tres.append(RES_LIST[-1])

        RULE_LIST = []
        FATHER_LIST = []
        FATHER_NAMES = []
        DEPTH_LIST = []
        COPY_NODE = {}
        HAS_COPY = {}
        ACTION = []

    # print the rules
    for p, x in enumerate(tres):
        for rule_idx in x['rule']:
            if rule_idx < 1000000:
                print(REVERSE_RULES_DICT[rule_idx], end=',')

    open('rulead%d.pkl' % which_10k, "wb").write(pickle.dumps(RULEAD))
    open('rule%d.pkl' % which_10k, "wb").write(pickle.dumps(RULES))
    open('process_datacopy%d.pkl' % which_10k, "wb").write(pickle.dumps(tres))

    exit(0)
