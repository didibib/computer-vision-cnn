import util
#TV-Human interaction data set
set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                 [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                 [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                 [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
                 
classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

# test set
test_set = [f'{classes[c]}_{i:04d}' for c in range(len(classes)) for i in set_1_indices[c]]
test_set_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
#print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
#print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

# training set
train_set = [f'{classes[c]}_{i:04d}' for c in range(len(classes)) for i in set_2_indices[c]]
train_set_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
#print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
#print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')

class_dict = util.create_dict_index(classes)
training_labels = util.convert_to_index(train_set_label, class_dict)
test_labels = util.convert_to_index(test_set_label, class_dict)