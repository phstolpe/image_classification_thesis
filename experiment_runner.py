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

with open('test.txt', 'a') as f:
    f.write('Dataset 1 \n\t Splits \n\t\t 80/20:')




for i in range(3):
    res = exp.run(i, splits[0])
    accs.append(res["accuracy"])
    with open('test.txt', 'a') as f:
        output = f'\n\t\t\t Iteration: {i}, Accuracy: {res["accuracy"]}'
        f.write(output)    

    #add average on last iteration
    if i == 2:
        avg = accs[0] + accs[1] + accs[2]
        print(avg)
        avg = avg/3
        print(avg)
        with open('test.txt', 'a' ) as f:
            output = f'\n\t\t Average: {avg}'
            f.write(output)






#köra tre gånger med tre olika eller samma seed?

#köra alla olika splits

