from genericpath import exists
import os
import subprocess

d4j_projects = []
# for i in range(1, 27):
#     d4j_projects.append("Chart-" + str(i))
# for i in range(1, 134):
#     d4j_projects.append("Closure-" + str(i))
# for i in range(1, 66):
#     d4j_projects.append("Lang-" + str(i))
# for i in range(1, 107):
#     d4j_projects.append("Math-" + str(i))
for i in range(1, 39):
    d4j_projects.append("Mockito-" + str(i))
# for i in range(1, 28):
#     d4j_projects.append("Time-" + str(i))

def run(bugid):
    print("Starting", bugid)
    cmd = ["python3", "testDefect4j.py", bugid]
    subp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.makedirs("emsemble-out", exist_ok=True)
    try:
        out = subp.stdout.decode('utf-8')
        err = subp.stderr.decode('utf-8')
        with open(f'emsemble-out/gen-{bugid}.log', 'w+') as f:
            f.write('stdout: '+out)
            f.write('stderr: '+err)
    except:
        with open(f'emsemble-out/gen-{bugid}.log', 'w+') as f:
            f.write('stderr: '+subp.stderr)
    print("Exited with code", subp.returncode)

for proj in d4j_projects:
    run(proj)

def run_reapir(bugid):
    print("Starting", bugid)
    cmd = ["python3", "repair.py", bugid]
    subp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.makedirs("emsemble-out", exist_ok=True)
    try:
        out = subp.stdout.decode('utf-8')
        err = subp.stderr.decode('utf-8')
        with open(f'emsemble-out/gen-{bugid}-repair.log', 'w+') as f:
            f.write('stdout: '+out)
            f.write('stderr: '+err)
    except:
        with open(f'emsemble-out/gen-{bugid}-repair.log', 'w+') as f:
            f.write('stderr: '+subp.stderr)
    print("Exited with code", subp.returncode)

for proj in d4j_projects:
    run_reapir(proj)