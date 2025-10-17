from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__=='__main__':
    lr_list = np.linspace(0.05,0.15,num=16-5).tolist()
    root_folder = Path('../results_maskdp_cifar10_eps4.0_20250826_170658/bs800/')

    for lr in tqdm(lr_list):
        print(root_folder / f'lr{lr:.2f}'.rstrip("0"))

        results_path = root_folder / f'lr{lr:.2f}'.rstrip("0")
        assert results_path.exists(), f'Results path {results_path} does not exist.'
        assert results_path.is_dir(), f'Results path {results_path} is not a directory.'


        accs_by_pruning_rate = {}


        for file in results_path.glob('**/summary.txt'):
            print(file)
            prune_rate = float(f'{file.parent.name.split('_')[-1]}')
            print(f'Prune Rate: {prune_rate:.2f}')
            with open(file, 'r') as f:
                first_line = f.readline()
                print(first_line)
                acc_string = first_line.split(': ')[-1]
                print(acc_string)
                acc_string =  acc_string[:-2] if acc_string.endswith("%\n") else acc_string
                print(acc_string)
                acc = float(acc_string)

                print(f'Accuracy: {acc}')
                accs_by_pruning_rate[prune_rate] = acc

        accs_by_pruning_rate = dict(sorted(accs_by_pruning_rate.items()))  # ascending keys

        plt.plot(accs_by_pruning_rate.keys(), accs_by_pruning_rate.values(), label=f'lr={lr}')

    plt.legend()
    plt.show()




