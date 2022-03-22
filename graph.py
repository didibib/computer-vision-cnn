import matplotlib.pyplot as plt
import pandas as pd
import json

def plot(filename):
    data = json.load(open(filename,))
    df = pd.DataFrame(data["fit_data"])
    print(df)
    
    fig, ax = plt.subplots()

    ax.plot(df['valid_loss'] ,label='valid_loss')
    ax.plot(df['valid_acc'] ,label='valid_acc')

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation")
    ax.legend(loc='best')
    plt.show()


plot("base_epoch_log.json")
plot("variant1_dense_epoch_log.json")
plot("variant2_kernel_epoch_log.json")
plot("variant3_lrelu_epoch_log.json")
plot("variant4_batch_norm_epoch_log.json")