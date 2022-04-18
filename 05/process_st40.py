# Stanford 40 images get resized and padded 

import util
from PIL import Image

# Stanford 40 data set
with open('st40_data/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    #print(f'Train files ({len(train_files)}):\n\t{train_files}')
    #print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

with open('st40_data/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    #print(f'Test files ({len(test_files)}):\n\t{test_files}')
    #print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
    
action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))

file = open('st40_data/actions.txt', 'w')
for label in action_categories:
    file.write(label+'\n')
file.close()

size = (164, 164)
l = lambda loc: util.resize_and_pad(Image.open('st40_data/images/'+ loc), size)
print('-- Processing images')
train_images = list(map(l, train_files))
test_images = list(map(l, test_files))

print('-- Saving images')
for i in range(len(train_images)):
    img = train_images[i]
    file = train_files[i]
    img.save('st40_data/proc_images/'+file)
    img.close()

for i in range(len(test_images)):
    img = test_images[i]
    file = test_files[i]
    img.save('st40_data/proc_images/'+file)
    img.close()