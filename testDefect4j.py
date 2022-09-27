# from ast import nodes
# from graphviz import Digraph

from copy import deepcopy
from repair import save_code_as_file
from run import *
from Searchnode import Node
from stringfycode import stringfyRoot
from tqdm import tqdm

import io
import javalang
import javalang.tree
import json
import numpy as np
import os
import pickle
import subprocess
import sys
import time


linenode = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement', 'ContinueStatement_ter', 'LocalVariableDeclaration',
            'condition', 'control', 'BreakStatement', 'ContinueStatement', 'ReturnStatement', "parameters", 'StatementExpression', 'return_type']

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 4"


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


n = 0


def setid(root):
    global n
    root.id = n
    n += 1
    for x in root.child:
        setid(x)


def solveLongTree(root, subroot) -> Tuple[Node, Dict[str, str], Dict[str, str]]:
    global n
    m = 'None'
    troot = 'None'
    for x in root.child:
        if x.name == 'name':
            m = x.child[0].name
    if len(root.getTreestr().strip().split()) >= 1000:
        tmp = subroot
        tree_str = tmp.getTreestr().strip()
        if len(tree_str.split()) >= 1000:
            print("ERROR!!!!!!! TOO LONG STATEMENT!!!!")
            return None, None, None
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
                ansroot.child.append(tmp.child[prenode])
                ansroot.num += tmp.child[prenode].getNum()
            if b:
                break
        troot = ansroot
    else:
        troot = root
    n = 0
    setid(troot)  # set id: root is 0, and increase the id by preorder traversal
    varnames = getLocVar(troot)
    fnum = -1
    vnum = -1
    vardic = {}
    vardic[m] = 'meth0'
    typedic = {}
    for x in varnames:
        if x[1].name == 'VariableDeclarator':
            vnum += 1
            vardic[x[0]] = 'loc' + str(vnum)
            t = -1
            for s in x[1].father.father.child:
                # print(s.name)
                if s.name == 'type':
                    t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert (t != -1)
            typedic[x[0]] = t
        else:
            fnum += 1
            vardic[x[0]] = 'par' + str(fnum)
            t = -1
            for s in x[1].child:
                if s.name == 'type':
                    t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert (t != -1)
            typedic[x[0]] = t
    return troot, vardic, typedic


def addter(root):
    if len(root.child) == 0:
        root.name += "_ter"
    for x in root.child:
        addter(x)
    return


def setProb(r: Node, p):
    r.possibility = p  # max(min(np.random.normal(0.8, 0.1, 10)[0], 1), 0)
    for x in r.child:
        setProb(x, p)


def getLineNode(root, block, add=True):
    ans = []
    block = block + root.name
    #print(root.name, 'lll')
    for x in root.child:
        if x.name in linenode:
            if 'info' in x.getTreestr() or 'assert' in x.getTreestr() or 'logger' in x.getTreestr() or 'LOGGER' in x.getTreestr() or 'system.out' in x.getTreestr().lower():
                continue
            x.block = block
            ans.append(x)
        else:
            # print(x.name)
            s = ""
            if not add:
                s = block
                #tmp = getLineNode(x, block)
            else:
                s = block + root.name
            #print(block + root.name + "--------")
            tmp = getLineNode(x, block)
            '''if x.name == 'then_statement' and tmp == []:
        print(tmp)
        print(x.father.printTree(x.father))
        assert(0)'''
            ans.extend(tmp)
    return ans


def getroottree(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def ismatch(root, subroot):
    index = 0
    #assert(len(subroot.child) <= len(root.child))
    #print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while index < len(root.child) and root.child[index].name != x.name:
            index += 1
        if index == len(root.child):
            return False
        if not ismatch(root.child[index], x):
            return False
        index += 1
    return True


def findSubtree(root, subroot):
    if root.name == subroot.name:
        if ismatch(root, subroot):
            return root
    for x in root.child:
        tmp = findSubtree(x, subroot)
        if tmp:
            return tmp
    return None


def generateAST(tree):
    sub = []
    if not tree:
        return ['None', '^']
    if isinstance(tree, str):
        tmpStr = tree
        tmpStr = tmpStr.replace(" ", "").replace(":", "")
        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
            tmpStr = "<string>"
        if len(tmpStr) == 0:
            tmpStr = "<empty>"
        if tmpStr[-1] == "^":
            tmpStr += "<>"
        sub.append(tmpStr)
        sub.append("^")
        return sub
    if isinstance(tree, list):
        if len(tree) == 0:
            sub.append("empty")
            sub.append("^")
        else:
            for ch in tree:
                subtree = generateAST(ch)
                sub.extend(subtree)
        return sub
    position = None
    if hasattr(tree, 'position'):
        # assert(0)
        position = tree.position
    curr = type(tree).__name__
    # print(curr)
    if True:
        if False:
            assert (0)  # sub.append((str(getLiteral(tree.children)))
        else:
            sub.append((curr, position))
            try:
                for x in tree.attrs:
                    if x == "documentation":
                        continue
                    if not getattr(tree, x):
                        continue
                    '''if x == 'prefix_operators':
                        node = getattr(tree, x)
                        print(type(node))
                        print(len(node))
                        print(node[0])
                        assert(0)
                    if type(getattr(tree, x)).__name__ not in nodes:
                        print(type(getattr(tree, x)).__name__)
                        continue'''
                    sub.append(x)
                    node = getattr(tree, x)
                    if isinstance(node, list):
                        if len(node) == 0:
                            sub.append("empty")
                            sub.append("^")
                        else:
                            for ch in node:
                                subtree = generateAST(ch)
                                sub.extend(subtree)
                    elif isinstance(node, javalang.tree.Node):
                        subtree = generateAST(node)
                        sub.extend(subtree)
                    elif not node:
                        continue
                    elif isinstance(node, str):
                        tmpStr = node
                        tmpStr = tmpStr.replace(" ", "").replace(":", "")
                        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
                            tmpStr = "<string>"
                        if len(tmpStr) == 0:
                            tmpStr = "<empty>"
                        if tmpStr[-1] == "^":
                            tmpStr += "<>"
                        sub.append(tmpStr)
                        sub.append("^")
                    elif isinstance(node, set):
                        for ch in node:
                            subtree = generateAST(ch)
                            sub.extend(subtree)
                    elif isinstance(node, bool):
                        sub.append(str(node))
                        sub.append("^")
                    else:
                        print(type(node))
                        assert (0)
                    sub.append("^")
            except AttributeError:
                assert (0)
                pass
        sub.append('^')
        return sub
    else:
        print(curr)
    return sub


'''def setProb(root, subroot, prob):
    root.possibility = max(min(max(root.possibility, prob), 0.98), 0.01)
    index = 0
    assert(len(subroot.child) <= len(root.child))
    #print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while root.child[index].name != x.name:
            #print(root.child[index].name, x.name)
            index += 1
        setProb(root.child[index], x, prob)
        index += 1'''


def getSubroot(treeroot: Node):
    currnode = treeroot
    lnode = None
    mnode = None
    while currnode:
        if currnode.name in linenode:
            lnode = currnode
            break
        currnode = currnode.father
    currnode = treeroot
    while currnode:
        if currnode.name == 'MethodDeclaration' or currnode.name == 'ConstructorDeclaration':
            mnode = currnode
            break
        currnode = currnode.father
    return lnode, mnode


def repair(treeroot, troot, oldcode, filepath, filepath2, patchpath, patchnum, isIf, mode, subroot, vardic, typedic, idxs, testmethods, idss, classname):
    global aftercode
    global precode
    actionlist = solveone(troot.printTreeWithVar(troot, vardic), troot.getTreeProb(
        troot), model, subroot, vardic, typedic, idxs, idss, classname, mode)
    for x in actionlist:
        if x.strip() in patchdict:
            continue
        #print('-', x)
        patchdict[x.strip()] = 1
        # print(x.split())
        root = getroottree(x.split())
        code = stringfyRoot(root, isIf, mode)
        # print(oldcode)
        print(precode[-1000:])
        print(code)
        print(aftercode[:1000])
        #copycode = deepcopy(liness)
        #copycode[lineid - 1] = code
        lnum = 0
        for x in code.splitlines():
            if x.strip() != "":
                lnum += 1
            else:
                continue
        print('lnum', lnum, mode)
        if lnum == 1 and 'if' in code:
            if mode == 0:
                continue
            afterlines = aftercode.splitlines()
            lnum = 0
            rnum = 0
            for p, x in enumerate(afterlines):
                if '{' in x:
                    lnum += 1
                if '}' in x:
                    if lnum == 0:
                        aftercode = "\n".join(
                            afterlines[:p] + ['}'] + afterlines[p:])
                        # print(aftercode)
                        # assert(0)
                        break
                    lnum -= 1
            tmpcode = precode + "\n" + code + aftercode
            tokens = javalang.tokenizer.tokenize(tmpcode)
            parser = javalang.parser.Parser(tokens)
        else:
            tmpcode = precode + "\n" + code + aftercode
            tokens = javalang.tokenizer.tokenize(tmpcode)
            parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse()
        except:
            # assert(0)
            print(code)
            continue
        open(filepath2, "w").write(tmpcode)
        bugg = False
        for t in testmethods:
            cmd = 'defects4j test -w buggy2/ -t %s' % t.strip()
            Returncode = ""
            child = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
            while_begin = time.time()
            while True:
                Flag = child.poll()
                print(Flag)
                if Flag == 0:
                    Returncode = child.stdout.readlines()  # child.stdout.read()
                    break
                elif Flag != 0 and time.time() - while_begin > 10:
                    child.kill()
                    break
                else:
                    time.sleep(1)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                continue
            else:
                bugg = True
                break
        if not bugg:
            print('success')
            patchnum += 1
            wf = open(patchpath + 'patch' + str(patchnum) + ".txt", 'w')
            wf.write(filepath + "\n")
            wf.write("-" + oldcode + "\n")
            wf.write("+" + code + "\n")
            if patchnum >= 5:
                return patchnum
    return patchnum


def getNodeById(root, line):
    if root.position:
        if root.position.line == line and root.name != 'IfStatement' and root.name != 'ForStatement':
            return root
    for x in root.child:
        t = getNodeById(x, line)
        if t:
            return t
    return None


def containID(root):
    ans = []
    if root.position is not None:
        ans.extend([root.position.line])
    for x in root.child:
        ans.extend(containID(x))
    return ans


def getAssignMent(root):
    if root.name == 'Assignment':
        return root
    for x in root.child:
        t = getAssignMent(x)
        if t:
            return t
    return None


def isAssign(line):
    #sprint(4, line.getTreestr())
    if 'Assignment' not in line.getTreestr():
        return False
    anode = getAssignMent(line)
    if anode.child[0].child[0].name == 'MemberReference' and anode.child[1].child[0].name == 'MethodInvocation':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
            v = anode.child[1].child[0].child[0].child[0].name
        except:
            return False
        print(m, v)
        return m == v
    if anode.child[0].child[0].name == 'MemberReference':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
        except:
            return False
        if "qualifier " + m in anode.child[1].getTreestr():
            return True
    return False
    #lst = line.split("=")
    #print(lst[0].split()[-1], lst[1])
    # return lst[0].split()[-1].strip() in lst[1].strip()


def get_method_range(tree: javalang.tree.CompilationUnit, mnode: Node, line_no: int) -> Tuple[str, int, int]:
    func_list = list()
    found_method = False
    result = containID(mnode)
    start_line = min(result)
    end_line = max(result)
    last_node = None
    for func, node in tree.filter(javalang.tree.MethodDeclaration):
        if start_line <= node.position.line <= end_line:
            print(node.name)
            print(node.position)
            print("FOUND IT!")
            found_method = True
            last_node = node
            break
    if found_method:
        return last_node.name, start_line, end_line
    last_node = None
    for func, node in tree.filter(javalang.tree.ConstructorDeclaration):
        if start_line <= node.position.line <= end_line:
            print(node.name)
            print(node.position)
            print("FOUND IT!")
            found_method = True
            last_node = node
            break
    if found_method:
        return last_node.name, start_line, end_line
    print("CANNOT FOUND FUNCTION LOCATION!!!!")
    return "0no_function_found", 0, 0


PROJECTS_V1_2 = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']
IDS_V1_2 = [
    list(range(1, 27)),
    list(range(1, 134)),
    list(range(1, 66)),
    list(range(1, 107)),
    list(range(1, 39)),
    list(range(1, 28)),
]

decoder_model = test()

user_given_bug_id = sys.argv[1]

# import pdb; pdb.set_trace()

project_names_list = [user_given_bug_id.split("-")[0]]
ids_list = [[int(user_given_bug_id.split("-")[1])]]

for i, project_name in enumerate(project_names_list):
    for idx in ids_list[i]:
        bug_id = project_name + "-" + str(idx)

        if bug_id != user_given_bug_id:
            continue

        print('p')

        x = project_name
        locationdir = 'location/groundtruth/%s/%d' % (x.lower(), idx)
        if not os.path.exists(locationdir):
            continue
        os.makedirs(f"buggy", exist_ok=True)
        os.system('defects4j checkout -p %s -v %db -w buggy/%s' %
                  (x, idx, bug_id))  # os.system('defects4j')

        patchnum = 0

        '''
        s = os.popen('defects4j export -p classes.modified -w buggy').readlines()
        if len(s) != 1:
            continue
        s = s[-1]
        '''

        lines = open(locationdir, 'r').readlines()
        location = []
        locationdict = {}
        for loc in lines:
            loc = loc.split("||")[0]
            classname, lineid = loc.split(':')
            classname = ".".join(classname.split(".")[:-1])
            location.append((classname, 1, eval(lineid)))
        dirs = os.popen(
            f'defects4j export -p dir.src.classes -w buggy/{bug_id}').readlines()[-1]

        # correctpath = os.popen('defects4j export -p classes.modified -w fixed').readlines()[-1]
        # fpath = "fixed/%s/%s.java"%(dirs, correctpath.replace('.', '/'))
        # fpathx = "buggy/%s/%s.java"%(dirs, correctpath.replace('.', '/'))
        # testmethods = os.popen('defects4j export -w buggy -p tests.trigger').readlines()
        '''
        wf = open(patchpath + 'correct.txt', 'w')
        wf.write(fpath + "\n")
        wf.write("".join(os.popen('diff -u %s %s'%(fpath, fpathx)).readlines()) + "\n")
        wf.close()
        '''

        data = []
        func_map: Dict[str, List[dict]] = dict()
        for j in range(len(location)):
            if j >= len(location):
                break
            patchdict = {}
            ac = location[j]
            classname = ac[0]
            fl_score = ac[1]
            if '$' in classname:
                classname = classname[:classname.index('$')]
            s = classname
            print('path', s)
            filepath = f"buggy/{bug_id}/{dirs}/{s.replace('.', '/')}.java"
            filepathx = "fixed/%s/%s.java" % (dirs, s.replace('.', '/'))
            try:
                lines1 = open(filepath, "r").read().strip()
            except:
                with open(filepath, "r", encoding="iso-8859-1") as f:
                    lines1 = f.read().strip()
            liness = lines1.splitlines()
            tokens = javalang.tokenizer.tokenize(lines1)
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse()
            tmproot = getroottree(generateAST(tree))
            lineid = ac[2]
            # current node by line number
            currroot = getNodeById(tmproot, lineid)
            # Subroot from curroot
            lnode, mnode = getSubroot(currroot)
            if mnode is None:
                continue
            funcname, startline, endline = get_method_range(
                tree, mnode, lineid)
            if filepath not in func_map:
                func_map[filepath] = list()
            func_map[filepath].append(
                {"function": funcname, "begin": startline, "end": endline})
            oldcode = liness[ac[2] - 1]
            isIf = True
            subroot = lnode     # line root
            treeroot = mnode    # method decl
            presubroot = None
            aftersubroot = None
            # print(treeroot.printTreeWithLine(treeroot))
            linenodes = getLineNode(treeroot, "")
            # print(lineid, 2)
            if subroot not in linenodes:
                # print(treeroot.getTreestr(), subroot.getTreestr())
                # if j == 19:
                #     assert(0)
                # print(j, subroot, '3')
                continue
            currid = linenodes.index(subroot)   # index of linenode in treeroot
            if currid > 0:
                presubroot = linenodes[currid - 1]  # previous root
            if currid < len(linenodes) - 1:
                aftersubroot = linenodes[currid + 1]  # after root
            setProb(treeroot, 2)
            addter(treeroot)
            if subroot is None:
                continue
            # print(lineid, 3, liness[lineid - 1], subroot.getTreestr(), len(data))
            # print(treeroot.printTreeWithLine(subroot))
            if True:  # 2: treeroot, 1: subroot, 3: prev, 4: after
                setProb(treeroot, 2)
                if subroot is not None:
                    setProb(subroot, 1)
                if aftersubroot is not None:
                    setProb(aftersubroot, 4)
                if presubroot is not None:
                    setProb(presubroot, 3)
                # print(containID(subroot))
                # range of subroot statement's line number
                cid = set(containID(subroot))
                maxl = -1
                minl = 1e10
                for l in cid:
                    maxl = max(maxl, l - 1)
                    minl = min(minl, l - 1)
                # print(maxl, liness[maxl + 1])
                precode = "\n".join(liness[0:minl])
                aftercode = "\n".join(liness[maxl + 1:])
                oldcode = "\n".join(liness[minl:maxl + 1])
                # troot: treeroot
                # vardic: variable dict
                # typedic: type of variables
                troot, vardic, typedic = solveLongTree(treeroot, subroot)
                if troot is None:
                    continue
                data.append({'bugid': user_given_bug_id, 'treeroot': treeroot, 'troot': troot, 'oldcode': oldcode,
                             'filepath': filepath, 'subroot': subroot, 'vardic': vardic,
                             'typedic': typedic, 'idss': bug_id, 'classname': classname,
                             'precode': precode, 'aftercode': aftercode, 'tree': troot.printTreeWithVar(troot, vardic),
                             'prob': troot.getTreeProb(troot), 'mode': 0, 'line': lineid, 'isa': False, 'fl_score': fl_score})
                # patchnum = repair(treeroot, troot, oldcode, filepath, filepath2, patchpath, patchnum, isIf, 0, subroot, vardic, typedic, idxs, testmethods, idss, classname)

        os.makedirs(f"d4j/{user_given_bug_id}", exist_ok=True)

        with open(f"d4j/{user_given_bug_id}/func_loc.json", "w") as f:
            json.dump(func_map, f, indent=4)

        print(data)

        ans = solveone(data, decoder_model)

        with open(f"d4j/{user_given_bug_id}/{user_given_bug_id}.json", "w") as f:
            json.dump(ans, f, indent=4)

        # save_code_as_file("./d4j", bugid, ans, func_map)
