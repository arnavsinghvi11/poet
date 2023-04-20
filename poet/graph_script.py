import subprocess
import matplotlib.pyplot as plt

models = ['resnet18', 'resnet18_patch', 'resnet18_fused', 'resnet18_inplace', 'resnet18_pretrained']
ram_consumptions = []

for model in models:
    output_file = f'{model}_output.txt'
    with open(output_file, 'w') as f:
        cmd = ['python', 'poet/solve.py', '--model', model, '--platform', 'a72', '--ram-budget', '3000000', '--runtime-budget', '7.6']
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                output = p.stdout.readline()
                if not output:
                    break
                print(output.decode().strip())
                f.write(output.decode())
                if 'Total RAM consumption of forward pass:' in output.decode():
                    ram_consumptions.append(int(output.decode().split(':')[-1].strip().split(' ')[0]))

            p.wait()
        except subprocess.CalledProcessError as e:
            print(f'Error running command: {e}')
            continue

plt.bar(models, ram_consumptions)
plt.xlabel('ResNet18 Models')
plt.ylabel('RAM Consumption (bytes)')
plt.title('RAM Consumption of ResNet18 Models')
plt.savefig('ram_consumptions.png')
plt.show()
