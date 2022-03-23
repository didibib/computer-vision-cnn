# Standalone file to create and save graphs from json data
import matplotlib.pyplot as plt
import pandas as pd
import json

def plot(filename):
    data = json.load(open('json/' + filename,))
    dftrain = pd.DataFrame(data['train_data'])
    
    fig, ax = plt.subplots()

    ax.plot(dftrain['val_loss'], label='val_loss')
    ax.plot(dftrain['val_acc'], label='val_acc')
    ax.plot(dftrain['train_loss'], label='train_loss')
    ax.plot(dftrain['train_acc'], label='train_acc')

    ax.set_xlabel("epoch")
    ax.set_ylabel("acc & loss")
    ax.legend(loc='best')
    plt.savefig('graphs/' + filename + '.jpg', bbox_inches='tight')
    plt.show()

plot("base_epoch_log.json")
plot("variant1_dense_epoch_log.json")
plot("variant2_kernel_epoch_log.json")
plot("variant3_lrelu_epoch_log.json")
plot("variant4_batch_norm_epoch_log.json")