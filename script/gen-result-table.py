from typing import List, Set, Dict, Tuple
import multiprocessing as mp
import os
import sys
import json

def check_dir(dir: str) -> bool:
    if not os.path.exists(dir):
        print(f"Dir {dir} does not exist!")
        return False
    return True

def get_correct_patch(correct_patch_csv: str) -> Dict[str, List[str]]:
    result = dict()
    with open(correct_patch_csv, "r") as f:
        for line in f.readlines():
            ln = line.strip()
            if len(ln) < 1 or ln.startswith("#"):
                continue
            tokens = ln.split(",")
            # print(tokens)
            # print(f"{len(bugid)} == {len(tokens[0])}")
            bugid = tokens[0]
            patches = tokens[1:]
            result[bugid] = patches
    return result

def sort_bugids(bugids: List[str]) -> List[str]:
    proj_dict = dict()
    for bugid in bugids:
        proj, id = bugid.split("-")
        if proj not in proj_dict:
            proj_dict[proj] = list()
        proj_dict[proj].append(int(id))
    projs = sorted(list(proj_dict.keys()))
    result = list()
    for proj in projs:
        ids = proj_dict[proj]
        ids.sort()
        for id in ids:
            result.append(f"{proj}-{id}")
    return result

def analyze_result(recoder_path: str, bugid: str, target_dir: str, correct_patches: List[str]) -> Tuple[int, float, int, float, int]:
    """
    Analyze the result of a bugid.
    Return: (plau iter, plau time, correct iter, correct time, total plau)
    """
    result_file = os.path.join(target_dir, f"msv-result.json")
    sim_file = os.path.join(recoder_path, "sim", bugid, f"{bugid}-sim.json")
    sim = dict()
    with open(sim_file, "r") as f:
        sim = json.load(f)
    if not os.path.exists(result_file):
        return -1, -1, -1, -1, -1
    with open(result_file, "r") as f:
        root = json.load(f)
        global_time = 0.0
        pi = -1
        pt = -1
        ci = -1
        ct = -1
        found_correct = False
        found_plausible = False
        total_plau = 0
        for elem in root:
            iter = elem["iteration"]
            time = elem["time"]
            is_pass = elem["pass_result"]
            config = elem["config"][0]
            patch_id = f'{config["id"]}-{config["case_id"]}'
            patch_loc = config["location"]
            if patch_loc in sim:
                patch = sim[patch_loc]
                fail_time = patch["fail_time"]
                pass_time = patch["pass_time"]
                tm = fail_time + pass_time
                global_time += tm
                time = global_time
            if patch_id in correct_patches:
                if not found_correct:
                    ci = iter
                    ct = time
                    found_correct = True
            if is_pass:
                total_plau += 1
                if not found_plausible:
                    pi = iter
                    pt = time
                    found_plausible = True
        return pi, pt, ci, ct, total_plau

def main(args: List[str]) -> None:
    if len(args) < 4:
        print("Usage: gen-result-table.py <recoder-path> <outdir> <id>")
        return
    recoder_path = args[1]
    outdir = args[2]
    id = args[3]
    if not os.path.exists(outdir):
        print(f"Outdir {outdir} does not exist!")
        return
    correct_patch_dict = get_correct_patch(os.path.join(recoder_path, "data", "correct_patch.csv"))
    bugids = list(correct_patch_dict.keys())
    bugids = sort_bugids(bugids)
    result = [",guided,,,,,seapr,,,,,recoder,,,,,",
              ",plau_impv%,corr_impv%,plau_iter,corr_iter,total_plau,plau_impv%,corr_impv%,plau_iter,corr_iter,total_plau,plau_impv%,corr_impv%,plau_iter,corr_iter,total_plau"]
    t_result = [",guided,,,,,seapr,,,,,recoder,,,,,",
                ",plau_impv%,corr_impv%,plau_time,corr_time,total_plau,plau_impv%,corr_impv%,plau_time,corr_time,total_plau,plau_impv%,corr_impv%,plau_time,corr_time,total_plau"]
    total_guided = [0, 0]
    t_total_guided = [0, 0]
    total_seapr = [0, 0]
    t_total_seapr = [0, 0]
    total_recoder = [0, 0]
    t_total_recoder = [0, 0]
    for bugid in bugids:
        # os.makedirs(os.path.join("/root/project/Recoder/switch-info", bugid), exist_ok=True)
        # os.system(f"cp {os.path.join(recoder_path, 'd4j', bugid, 'switch-info.json')} {os.path.join('/root/project/Recoder/switch-info', bugid)}")
        # continue
        original_dir = os.path.join(outdir, f"{bugid}-recoder-{id}")
        if not check_dir(original_dir):
            continue
        result_recoder = analyze_result(recoder_path, bugid, original_dir, correct_patch_dict[bugid])
        p_base = result_recoder[0]
        t_p_base = result_recoder[1]
        c_base = result_recoder[2]
        t_c_base = result_recoder[3]
        plau_iter = "-"
        plau_time = "-"
        corr_iter = "-"
        corr_time = "-"
        plau_impv = "-"
        t_plau_impv = "-"
        corr_impv = "-"
        t_corr_impv = "-"
        if p_base > 0:
            plau_iter = str(p_base)
            plau_time = str(t_p_base)
            total_recoder[0] += p_base
            t_total_recoder[0] += t_p_base
        if c_base > 0:
            corr_iter = str(c_base)
            corr_time = str(t_c_base)
            total_recoder[1] += c_base
            t_total_recoder[1] += t_c_base
        result_str_recoder = f"-,-,{plau_iter},{corr_iter},{result_recoder[4]}"
        t_result_str_recoder = f"-,-,{plau_time},{corr_time},{result_recoder[4]}"
        seapr_dir = os.path.join(outdir, f"{bugid}-seapr-{id}")
        result_seapr = analyze_result(recoder_path, bugid, seapr_dir, correct_patch_dict[bugid])
        plau_seapr = result_seapr[0]
        t_plau_seapr = result_seapr[1]
        corr_seapr = result_seapr[2]
        t_corr_seapr = result_seapr[3]
        plau_iter = "-"
        plau_time = "-"
        corr_iter = "-"
        corr_time = "-"
        plau_impv = "-"
        t_plau_impv = "-"
        corr_impv = "-"
        t_corr_impv = "-"
        if plau_seapr > 0:
            plau_iter = str(plau_seapr)
            plau_time = str(t_plau_seapr)
            total_seapr[0] += plau_seapr
            t_total_seapr[0] += t_plau_seapr
            if p_base > 0:
                plau_impv = '%0.4f' % (100 * (p_base - plau_seapr) / p_base)
                t_plau_impv = '%0.4f' % (100 * (t_p_base - t_plau_seapr) / t_p_base)
        if corr_seapr > 0:
            corr_iter = str(corr_seapr)
            corr_time = str(t_corr_seapr)
            total_seapr[1] += corr_seapr
            t_total_seapr[1] += t_corr_seapr
            if c_base > 0:
                corr_impv = '%0.4f' % (100 * (c_base - corr_seapr) / c_base)
                t_corr_impv = '%0.4f' % (100 * (t_c_base - t_corr_seapr) / t_c_base)
        result_str_seapr = f"{plau_impv},{corr_impv},{plau_iter},{corr_iter},{result_seapr[4]}"
        t_result_str_seapr = f"{t_plau_impv},{t_corr_impv},{plau_time},{corr_time},{result_seapr[4]}"
        result_guided = list()
        cc = 0
        pc = 0
        gpi = 0
        gpt = 0
        gci = 0
        gct = 0
        total_plau_guided = 0
        tc = 0
        for i in range(20):
            guided_dir = os.path.join(outdir, f"{bugid}-guided-{id}-{i}")
            tmp_result = analyze_result(recoder_path, bugid, guided_dir, correct_patch_dict[bugid])
            if tmp_result[0] >= 0:
                gpi += tmp_result[0]
                gpt += tmp_result[1]
                pc += 1
            if tmp_result[2] >= 0:
                gci += tmp_result[2]
                gct += tmp_result[3]
                cc += 1
            if tmp_result[4] > 0:
                total_plau_guided += tmp_result[4]
                tc += 1
        plau_guided = "-"
        t_plau_guided = "-"
        plau_impv = "-"
        t_plau_impv = "-"
        corr_guided = "-"
        t_corr_guided = "-"
        corr_impv = "-"
        t_corr_impv = "-"
        if pc > 0:
            gpi /= pc
            gpt /= pc
            total_guided[0] += gpi
            t_total_guided[0] += gpt
            plau_guided = str(gpi)
            t_plau_guided = str(gpt)
            if p_base > 0:
                plau_impv = '%0.4f' % (100 * (p_base - gpi) / p_base)
                t_plau_impv = '%0.4f' % (100 * (t_p_base - gpt) / t_p_base)
        if cc > 0:
            gci /= cc
            gct /= cc
            total_guided[1] += gci
            t_total_guided[1] += gct
            corr_guided = str(gci)
            t_corr_guided = str(gct)
            if c_base > 0:
                corr_impv = '%0.4f' % (100 * (c_base - gci) / c_base)
                t_corr_impv = '%0.4f' % (100 * (t_c_base - gct) / t_c_base)
        if tc > 0:
            total_plau_guided /= tc
        
        result_str_guided = f"{plau_impv},{corr_impv},{plau_guided},{corr_guided},{total_plau_guided}"
        t_result_str_guided = f"{t_plau_impv},{t_corr_impv},{plau_guided},{corr_guided},{total_plau_guided}"
        result_str = f"{bugid},{result_str_guided},{result_str_seapr},{result_str_recoder}"        
        t_result_str = f"{bugid},{t_result_str_guided},{t_result_str_seapr},{t_result_str_recoder}"
        result.append(result_str)
        t_result.append(t_result_str)
    total_str = f"TOTAL,,,{total_guided[0]},{total_guided[1]},,,,{total_seapr[0]},{total_seapr[1]},,,,{total_recoder[0]},{total_recoder[1]},,"
    result.append(total_str)
    result.extend(['"pos","=COUNTIF(B$3:B$49, "">0"")","=COUNTIF(C$3:C$49, "">0"")"," ",," ","=COUNTIF(G$3:G$49, "">0"")","=COUNTIF(H$3:H$49, "">0"")"," ",," "," ","=COUNTIF(M$3:M$49, "">0"")"," ","=COUNTIF(O$3:O$49, "">0"")",',
                   '"neg","=COUNTIF(B$3:B$49, ""<0"")","=COUNTIF(C$3:C$49, ""<0"")"," ",," ","=COUNTIF(G$3:G$49, ""<0"")","=COUNTIF(H$3:H$49, ""<0"")"," ",," "," ","=COUNTIF(M$3:M$49, ""<0"")"," ","=COUNTIF(O$3:O$49, ""<0"")",',
                   '"eq","=COUNTIF(B$3:B$49, ""=0"")","=COUNTIF(C$3:C$49, ""=0"")"," ",," ","=COUNTIF(G$3:G$49, ""=0"")","=COUNTIF(H$3:H$49, ""=0"")"," ",," "," "," "," "," ",',
                   '"Pos 10%","=COUNTIF(B$3:B$49, "">10"")","=COUNTIF(C$3:C$49, "">10"")"," ",," ","=COUNTIF(G$3:G$49, "">10"")","=COUNTIF(H$3:H$49, "">10"")"," ",," "," "," "," "," ",',
                   '"Neg 10%","=COUNTIF(B$3:B$49, ""<-10"")","=COUNTIF(C$3:C$49, ""<-10"")"," ",," ","=COUNTIF(G$3:G$49, ""<-10"")","=COUNTIF(H$3:H$49, ""<-10"")"," ",," "," "," "," "," ",',
                   '"pos rate","=(SUMIF(B$3:B$49, "">0"", N$3:N$49) - SUMIF(B$3:B$49, "">0"", D$3:D$49)) / (SUMIF(B$3:B$49, "">0"", N$3:N$49))","=(SUMIF(C$3:C$49, "">0"", O$3:O$49) - SUMIF(C$3:C$49, "">0"", E$3:E$49)) / (SUMIF(C$3:C$49, "">0"", O$3:O$49))"," ",," ","=(SUMIF(G$3:G$49, "">0"", N$3:N$49) - SUMIF(G$3:G$49, "">0"", I$3:I$49)) / (SUMIF(G$3:G$49, "">0"", N$3:N$49))","=(SUMIF(H$3:H$49, "">0"", O$3:O$49) - SUMIF(H$3:H$49, "">0"", J$3:J$49)) / (SUMIF(H$3:H$49, "">0"", O$3:O$49))"," ",," "," "," "," "," ",',
                   '"neg rate","=(SUMIF(B$3:B$49, ""<0"", N$3:N$49) - SUMIF(B$3:B$49, ""<0"", D$3:D$49)) / (SUMIF(B$3:B$49, ""<0"", N$3:N$49))","=(SUMIF(C$3:C$49, ""<0"", O$3:O$49) - SUMIF(C$3:C$49, ""<0"", E$3:E$49)) / (SUMIF(C$3:C$49, ""<0"", O$3:O$49))"," ",," ","=(SUMIF(G$3:G$49, ""<0"", N$3:N$49) - SUMIF(G$3:G$49, ""<0"", I$3:I$49)) / (SUMIF(G$3:G$49, ""<0"", N$3:N$49))","=(SUMIF(H$3:H$49, ""<0"", O$3:O$49) - SUMIF(H$3:H$49, ""<0"", J$3:J$49)) / (SUMIF(H$3:H$49, ""<0"", O$3:O$49))"," ",," "," "," "," "," ",'])
    with open(os.path.join(outdir, f"recoder-result-{id}.csv"), "w") as f:
        for line in result:
            f.write(line + "\n")

if __name__ == "__main__":
    main(sys.argv)
