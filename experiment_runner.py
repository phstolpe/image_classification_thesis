from numpy import average
import experiment as exp
import time
import sys
import random

#percentages used for testing. the rest will be used for training 
splits = [0.2, 0.4, 0.6, 0.8]
output_file = sys.argv[1]
ds_path= sys.argv[2]

#clean file
open(output_file, 'w').close()

time_start = time.perf_counter()
with open(output_file, 'a') as f:
    dataset_info = f'Dataset: {ds_path}'
    f.write(dataset_info)

    #do all the splits for the dataset
    for i in range(len(splits)):
        split_info = f'\nPercentage of dataset used for testing: {splits[i]}'
        print(split_info)
        f.write(split_info)
        accs = []

        #runs every split three times and saves the accuracy to calculate average
        for j in range(3):
            seed = random.randint(1, 100)
            res = exp.run(splits[i], ds_path, seed)
            accuracy = res["accuracy"]
            accs.append(accuracy)
            output = f'\n\t Iteration: {j}, Seed used: {seed}, Accuracy: {accuracy:.3f}'
            print(output)
            f.write(output)    

            #calculate average on last iteration and write to output file
            if j == 2:
                avg = average(accs)
                output = f'\n Average: {avg:.3f}\n'
                f.write(output)

    time_end= time.perf_counter()
    duration = time_end - time_start
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    profile = f'\nExperiment took: {duration:.3f} seconds\nLocal time: {local_time}'
    f.write(profile)
