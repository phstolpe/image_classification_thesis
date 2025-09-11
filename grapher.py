import matplotlib.pyplot as plt
import numpy as np



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
    print('dataset 1')
    for i in range(len(p)):
        print(f'percentage: {p[i]} and average: {a[i]}')

    p2, a2 = extract_and_average(result_file2)
    print('dataset 2')
    for i in range(len(p2)):
        print(f'percentage: {p2[i]} and average: {a2[i]}')


    #y axis will be accuracy and x axis will be percentage
    plt.plot(p,a, marker='o', label='Dataset 1')
    plt.plot(p2,a2, marker='o', label='Dataset 2')
    plt.title('Dataset performance over different train/test splits')
    plt.xlabel('Percentage used for testing')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('result_graph.pdf')
    plt.show()


#results files
file1 = 'result1.txt'
file2 = 'result2.txt'

create_graph(file1, file2)

