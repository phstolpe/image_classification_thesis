import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def extract_and_average(filename):
    percentage_used_for_testing = []
    average_accuracy = []
    with open(filename, 'r')as f:
        for line in f:
            if line.strip().startswith('Percentage'):
                percentage = float(line.split(':')[1].strip())
                percentage_used_for_testing.append(percentage)
            if line.strip().startswith('Average'):
                acc = float(line.split(':')[1].strip())
                average_accuracy.append(acc)
    return percentage_used_for_testing, average_accuracy

def create_graph(result_file1, result_file2):
    p, a = extract_and_average(result_file1)

    p2, a2 = extract_and_average(result_file2)


    #y axis will be accuracy and x axis will be percentage
    plt.plot(p,a, marker='o', label='Dataset 1')
    plt.plot(p2,a2, marker='o', label='Dataset 2')
    plt.title('Dataset performance over different train/test splits')
    plt.xlabel('Percentage used for testing')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('result_graph.pdf')

def create_table(filename: str, title: str):
    data = []
    with open(filename, 'r')as f:
        for line in f:
            if(line.startswith('Percentage')):
                percentage_used_for_testing = float(line.split(':')[1].strip())
                data.append(percentage_used_for_testing)
            elif(line.strip().startswith('Iteration')):
                accuracy = float(line.split('Accuracy:')[1].strip())
                data.append(accuracy)
            elif(line.strip().startswith('Average')):
                average = float(line.split(':')[1].strip())
                data.append(average)

    data = np.array_split(data, 4)
    df = pd.DataFrame(data, columns = ['split', 'run_1', 'run_2', 'run_3', 'average'])
    df['std'] = df[['run_1', 'run_2', 'run_3']].std(axis=1)
    df = df.round(3)

    fig, ax = plt.subplots(figsize=(10,2))

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    # fig.tight_layout()
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1,1.5)
    
    plt.savefig(title + '_fig.pdf')

#results files
file1 = 'result1.txt'
file2 = 'result2.txt'


create_graph(file1, file2)
create_table(file1, 'Dataset 1')
create_table(file2, 'Dataset 2')

