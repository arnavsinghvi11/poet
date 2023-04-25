import subprocess
import sys
import os
import re
import matplotlib.pyplot as plt
from argparse import ArgumentParser

#run with the following command "python poet/graph_script.py --model_type "{model_type}" --optimizations "{optimizations, starts with empty space and follow with optimizations names delimited by commas}" 
if __name__ == "__main__":
    parser = ArgumentParser(description="Graphs")
    parser.add_argument("--model", type=str)
    parser.add_argument("--optimizations", type=str)
args = parser.parse_args()
optimizations = args.optimizations.split(",")
output_folder = "models"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ram_consumptions, runtimes, cpu_powers, memory_paging_powers = [], [], [], []
for i in range(-1, len(optimizations)):
    if i == -1:
        cmd = ["python", "poet/solve.py", "--model", args.model, "--platform", "a72", "--ram-budget", "30000000", "--runtime-budget", "10"]
        output_path = os.path.join(output_folder, f"{args.model}_output.txt")
    else:
        cmd = ["python", "poet/solve.py", "--model", args.model, "--platform", "a72", "--ram-budget", "30000000", "--runtime-budget", "10", "--optimization", optimizations[i]]
        output_path = os.path.join(output_folder, f"{optimizations[i]}_output.txt")
    with open(output_path, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            output = p.stdout.readline()
            if not output:
                break
            f.write(output.decode(sys.stdout.encoding))
            if "Total RAM consumption of forward pass" in output.decode(sys.stdout.encoding):
                ram_consumption = int(output.decode(sys.stdout.encoding).split(":")[1].strip().split()[0])
                ram_consumptions.append(ram_consumption)
            if "Total runtime of graph (forward + backward)" in output.decode(sys.stdout.encoding):
                runtime = float(re.findall("\d+\.\d+", output.decode(sys.stdout.encoding))[0])
                runtimes.append(runtime)
            if "POET successfully found an optimal solution with a memory budget" in output.decode(sys.stdout.encoding):
                cpu_power = float(re.findall("\d+\.\d+", output.decode(sys.stdout.encoding))[0])
                cpu_powers.append(cpu_power)
            if "POET successfully found an optimal solution with a memory budget" in output.decode(sys.stdout.encoding):
                memory_paging_power = float(re.findall("\d+\.\d+", output.decode(sys.stdout.encoding))[1])
                memory_paging_powers.append(memory_paging_power)
        p.wait()
optimizations.insert(0, 'ICML')
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(optimizations, ram_consumptions)
ax.set_xlabel(str(args.model) + " Models")
ax.set_ylabel("RAM Consumption (bytes)")
ax.set_title("RAM Consumption for " + str(args.model) + " Models")
plt.savefig(os.path.join(output_folder, "ram_consumptions.png"))
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(optimizations, runtimes)
ax.set_xlabel(str(args.model) + " Models")
ax.set_ylabel("Total Runtime (ms)")
ax.set_title("Total Runtime for " + str(args.model) + " Models")
plt.savefig(os.path.join(output_folder, "runtimes.png"))
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(optimizations, cpu_powers)
ax.set_xlabel(str(args.model) + " Models")
ax.set_ylabel("CPU Power (J)")
ax.set_title("CPU Power for " + str(args.model) + " Models")
plt.savefig(os.path.join(output_folder, "cpu_powers.png"))
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(optimizations, memory_paging_powers)
ax.set_xlabel(str(args.model) + " Models")
ax.set_ylabel("Memory Paging Power (J)")
ax.set_title("Memory Paging Power " + str(args.model) + " Models")
plt.savefig(os.path.join(output_folder, "memory_paging_powers.png"))
plt.show()