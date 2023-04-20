import subprocess

# Run the command and capture the output
models = ['resnet18', "resnet18_patch", "resnet18_fused", "resnet18_inplace","resnet18_pretrained"]

import subprocess
import sys

with open("output.txt", "w") as f:
    for m in models:
        cmd = ["python", "poet/solve.py", "--model", m, "--platform", "a72", "--ram-budget", "3000000", "--runtime-budget", "7.6"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        output = p.stdout.readline()
        if not output:
            break
        print(output.decode(sys.stdout.encoding), end="")
        f.write(output.decode(sys.stdout.encoding))

    p.wait()

