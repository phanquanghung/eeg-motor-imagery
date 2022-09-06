from braindecode.util import set_random_seeds, np_to_var, var_to_np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np
from numpy.random import RandomState
import pickle
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import mne

import util.shallow_net
from util.utilfunc import get_balanced_batches
from util.preproc import plot_confusion_matrix

cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)

subj = 1

for dataset in [BNCI2014001(), PhysionetMI(), Cho2017()]:
    data = dataset.get_data(subjects=[subj])
    ds_name = dataset.code
    ds_type = dataset.paradigm
    sess = 'session_T' if ds_name == "001-2014" else 'session_0'
    run = sorted(data[subj][sess])[0]
    ds_ch_names = data[subj][sess][run].info['ch_names']  # [0:22]
    ds_sfreq = data[subj][sess][run].info['sfreq']
    print("{} is an {} dataset, acquired at {} Hz, with {} electrodes\nElectrodes names: ".format(ds_name, ds_type, ds_sfreq, len(ds_ch_names)))
    print(ds_ch_names)
    print()

ds_src1 = Cho2017()
ds_src2 = PhysionetMI()
ds_tgt = BNCI2014001()

fmin, fmax = 4, 32
raw = ds_tgt.get_data(subjects=[1])[1]['session_T']['run_1']
tgt_channels = raw.pick_types(eeg=True).ch_names
sfreq = 250.
prgm_2classes = MotorImagery(n_classes=2, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
prgm_4classes = MotorImagery(n_classes=4, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)

X_src1, label_src1, m_src1 = prgm_2classes.get_data(dataset=ds_src1, subjects=[1, 2, 3])
X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=[1, 2, 3, 4, 5])
X_tgt, label_tgt, m_tgt = prgm_4classes.get_data(dataset=ds_tgt, subjects=[1])

print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))
print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
print("Target dataset has {} trials with {} electrodes and {} time samples".format(*X_tgt.shape))

print ("\nSource dataset 1 include labels: {}".format(np.unique(label_src1)))
print ("Source dataset 2 include labels: {}".format(np.unique(label_src2)))
print ("Target dataset 1 include labels: {}".format(np.unique(label_tgt)))

def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    else: return 2


y_src1 = np.array([relabel(l) for l in label_src1])
y_src2 = np.array([relabel(l) for l in label_src2])
y_tgt = np.array([relabel(l) for l in label_tgt])

print("Only right-/left-hand labels are used and first source dataset does not have other labels:")
print(np.unique(y_src1), np.unique(y_src2), np.unique(y_tgt))

window_size = min(X_src1.shape[2], X_src2.shape[2], X_tgt.shape[2])

X_train = np.concatenate((X_src1[:, :, :window_size], X_src2[:, :, :window_size], X_tgt[:100, :, :window_size]))
y_train = np.concatenate((y_src1, y_src2, y_tgt[:100]))

X_val = X_tgt[100:150, :, :window_size]
y_val = y_tgt[100:150]

X_test = X_tgt[150:, :, :window_size]
y_test = y_tgt[150:]

print("Train:  there are {} trials with {} electrodes and {} time samples".format(*X_train.shape))
print("\nValidation: there are {} trials with {} electrodes and {} time samples".format(*X_val.shape))
print("\nTest: there are {} trials with {} electrodes and {} time samples".format(*X_test.shape))

class TrainObject(object):
    def __init__(self, X, y):
        assert len(X) == len(y)
        # Normalised, you could choose other normalisation strategy
        mean = np.mean(X,axis=1,keepdims=True)
        # here normalise across channels as an example, unlike the in the sleep kit
        std = np.std(X, axis=1, keepdims=True)
        X = (X - mean) / std
        # we scale it to 1000 as a better training scale of the shallow CNN
        # according to the orignal work of the paper referenced above
        self.X = X.astype(np.float32) * 1e3
        self.y = y.astype(np.int64)

train_set = TrainObject(X_train, y=y_train)
valid_set = TrainObject(X_val, y=y_val)
test_set = TrainObject(X_test, y=y_test)

in_chans = X_train.shape[1]
labelsize = len(np.unique(y_train))
model = util.shallow_net.EEGShallowClassifier(in_chans, labelsize, window_size, return_feature=False)
if cuda:
    model.cuda()


batch_size = 60
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.5*0.001)

total_epoch = -1
Tlosses, Taccuracies = [], []
Vlosses, Vaccuracies = [], []
highest_acc = 0

savename = "test.pth"

start=time.time()

for i_epoch in range(15):
    total_epoch += 1
    # Randomize batches ids and get iterater 'i_trials_in_batch'
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                             batch_size=batch_size)
    # Set model to training mode
    model.train()
    for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X for training
        batch_X = train_set.X[i_trials][:, :, :, None]
        batch_y = train_set.y[i_trials]
        # convert from nparray to torch tensor
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        # Compute outputs of the network
        outputs = model(net_in)
        # Compute the loss
        loss = F.nll_loss(outputs, net_target)
        # Do the backpropagation
        loss.backward()
        # Update parameters with the optimizer
        optimizer.step()
    # Set model to evaluation mode
    model.eval()
    print("Epoch {:d}".format(total_epoch))
    average_acc = []
    average_loss = []
    
    # Here we compute training accuracy and validation accuracy of current model
    for setname, dataset in (('Train', train_set), ('Valid', valid_set)):
        i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False,
                                                 batch_size=60)
        outputs=None
        for i_trials in i_trials_in_batch:
            batch_X = dataset.X[i_trials][:, :, :, None]
            batch_y = dataset.y[i_trials]
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            toutputs = model(net_in)
            if outputs is None:
                temp = toutputs.cpu()
                outputs = temp.detach().numpy()
            else:
                temp = toutputs.cpu()
                outputs = np.concatenate((outputs,temp.detach().numpy()))
        net_target = np_to_var(dataset.y)
        loss = F.nll_loss(torch.from_numpy(outputs), net_target)
        print("{:6s} Loss: {:.5f}".format(
            setname, float(var_to_np(loss))))
        predicted_labels = np.argmax((outputs), axis=1)
        accuracy = np.mean(dataset.y  == predicted_labels)
        
        print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))
        if setname == 'Train':
            Tlosses.append(loss)
            Taccuracies.append(accuracy)
            current_Tacc=accuracy
        elif setname == 'Valid':
            Vlosses.append(loss)
            Vaccuracies.append(accuracy)
            if accuracy>=highest_acc:
                torch.save({
                    'in_chans': in_chans,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'n_classes': 3,
                    'input_time_length': window_size
                }, savename)

                highest_acc=accuracy
                print('model saved')
                # plot_confusion_matrix(dataset.y, predicted_labels, 
                #                       classes=['LH', 'RH', 'Other'], normalize=True,
                #                       title='Validation confusion matrix')
                # plt.show()
        else:
            average_acc.append(accuracy)
            average_loss.append(accuracy)
end = time.time()

print('time is {}'.format(end-start))

model = util.shallow_net.EEGShallowClassifier(in_chans, labelsize, window_size, return_feature=False)
if cuda:
    model.cuda()
checkpoint = torch.load(savename)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

average_acc, average_loss = [], []
setname = 'testset'
dataset = test_set

i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False,
                                         batch_size=30)
outputs=None
for i_trials in i_trials_in_batch:
    # Have to add empty fourth dimension to X
    batch_X = dataset.X[i_trials][:, :, :, None]
    batch_y = dataset.y[i_trials]
    net_in = np_to_var(batch_X)
    if cuda:
        net_in = net_in.cuda()
    toutputs = model(net_in)
    if outputs is None:
        temp = toutputs.cpu()
        outputs = temp.detach().numpy()
    else:
        temp = toutputs.cpu()
        outputs = np.concatenate((outputs,temp.detach().numpy()))

net_target = np_to_var(dataset.y)
loss = F.nll_loss(torch.from_numpy(outputs), net_target)
print("{:6s} Loss: {:.5f}".format(setname, float(var_to_np(loss))))
predicted_labels = np.argmax((outputs), axis=1)
accuracy = np.mean(dataset.y  == predicted_labels)

print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))
# plot_confusion_matrix(dataset.y, predicted_labels, 
#                       classes=['LH','RH','Other'], normalize=True,
#                       title='Validation confusion matrix')
# plt.show()