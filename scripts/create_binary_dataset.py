import shutil
import os

#first dataset
source_dir = '../fresh-and-stale-images/'
target_dir = '../processed_datasets/dataset1/' 

rotten = target_dir+'rotten/'
fresh = target_dir+'fresh/'


os.makedirs(rotten ,exist_ok=True)
os.makedirs(fresh ,exist_ok=True)

sum = 0
for folder in os.listdir(source_dir):
    path = os.path.join(source_dir, folder)
    if not os.path.isdir(path):
            continue
    sum+= len(os.listdir(path))
    if folder.startswith('f'):
        for file in os.listdir(path):
            src_file = os.path.join(path, file)
            dst = fresh
            shutil.copy(src_file, dst)
    if folder.startswith('s'):
        for file in os.listdir(path):
            src_file = os.path.join(path, file)
            dst = rotten
            shutil.copy(src_file, dst)

rsum = len(os.listdir(rotten))
fsum = len(os.listdir(fresh))
print(f"sum: {sum} and rotten: {rsum} and fresh: {fsum} and total{fsum+rsum}")

#second dataset
source_dir = '../FRUIT-16K/'
target_dir = '../processed_datasets/dataset2/' 

rotten2 = target_dir+'rotten/'
fresh2 = target_dir+'fresh/'


os.makedirs(rotten2, exist_ok=True)
os.makedirs(fresh2, exist_ok=True)

sum2 = 0
for folder in os.listdir(source_dir):
    path = os.path.join(source_dir, folder)
    if not os.path.isdir(path):
            continue
    sum2+= len(os.listdir(path))
    if folder.startswith('F'):
        for file in os.listdir(path):
            src_file = os.path.join(path, file)
            new_filename = 'f_'+folder[2:]+'_'+file
            dst = os.path.join(fresh2, new_filename)
            shutil.copy(src_file, dst)
    if folder.startswith('S'):
        for file in os.listdir(path):
            src_file = os.path.join(path, file)
            new_filename = 'r_'+folder[2:]+'_'+file
            dst = os.path.join(rotten2, new_filename)
            shutil.copy(src_file, dst)

rsum2 = len(os.listdir(rotten2))
fsum2 = len(os.listdir(fresh2))
print(f"sum: {sum2} and rotten: {rsum2} and fresh: {fsum2} and total{fsum2+rsum2}")
