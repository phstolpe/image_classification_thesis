from numpy import average
import experiment as exp

#clean file
open('test.txt', 'w').close()

#percentages used for training. the rest will be used for testing accuracy
splits = [0.2, 0.4, 0.6, 0.8]
paths_to_ds = ['./processed_datasets/dataset1/', './processed_datasets/dataset2/']

for k in range(len(paths_to_ds)):
    print(f'dataset {k+1}')
    with open('test.txt', 'a')as f:
        dataset_info = f'Dataset: {k+1}'
        f.write(dataset_info)
    #do all the splits for a certain dataset
    for i in range(len(splits)):
        with open('test.txt', 'a') as f:
            split_info = f'\n\tPercentage used for training: {splits[i]}'
            print(split_info)
            f.write(split_info)
        accs = []

        #runs every split three times and saves the accuracy to calculate average
        for j in range(3):
            res = exp.run(j, splits[i], paths_to_ds[k])
            accuracy = res["accuracy"]
            accs.append(accuracy)
            with open('test.txt', 'a') as f:
                output = f'\n\t\t Iteration: {j}, Accuracy: {accuracy}'
                print(output)
                f.write(output)    

            #calculate average on last iteration and write to output file
            if j == 2:
                avg = accs[0] + accs[1] + accs[2]
                avg = avg/3.0
                with open('test.txt', 'a' ) as f:
                    output = f'\n\t Average: {avg}\n'
                    f.write(output)

