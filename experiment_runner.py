import json

from numpy import average
import experiment as exp




#clean file
open('test.txt', 'w').close()

#percentages used for training. the rest will be used for testing accuracy
splits = [0.2, 0.4, 0.6, 0.8]
tests = []
accs = []
results = {}


#json
# results["dataset 1"] = {
#         "splits": [
#             "0.2": {
#                 "tests": [{
#
#             }]
#
#
#         }
#
#
#     ]}




for i in range(len(splits)):
    split_info = f'\n\t Splits \n\t\t {splits[i]}:'

    with open('test.txt', 'a') as f:
        f.write(split_info)

    for j in range(3):
        res = exp.run(j, splits[0])
        accs.append(res["accuracy"])
        with open('test.txt', 'a') as f:
            output = f'\n\t\t\t Iteration: {j}, Accuracy: {res["accuracy"]}'
            f.write(output)    

        #add average on last iteration
        if j == 2:
            avg = accs[0] + accs[1] + accs[2]
            print(avg)
            avg = avg/3
            print(avg)
            with open('test.txt', 'a' ) as f:
                output = f'\n\t\t Average: {avg}'
                f.write(output)






#köra tre gånger med tre olika eller samma seed?

#köra alla olika splits

