import json
import matplotlib.pyplot as plt



def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def main():
    experiment_folder = './output/2020_05_23_15_10_44'
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

    plt.plot(
        [x['iteration'] for x in experiment_metrics],
        [x['total_loss'] for x in experiment_metrics])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()