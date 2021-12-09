import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

import syft as sy
from Data_utils_crypto import split_genSpoof,Dataset_ASVspoof2019_LA

hook = sy.TorchHook(torch)
client = sy.VirtualWorker(hook, id="client")
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

device = 'cpu'

class Arguments():
    def __init__(self):
        self.batch_size = 32
        self.test_batch_size = 8
        self.epochs = 100
        self.weight_decay = 0.0003
        self.lr = 0.0001

args = Arguments()

d_label_trn,utt_train_path,_ = split_genSpoof(dir_meta = '/home/crypto/protocols/ASVspoof2019.LA.cm.train.trn.txt')
d_label_dev, l_dev, attack_id_dev = split_genSpoof(dir_meta = '/home/crypto/protocols/ASVspoof2019.LA.cm.dev.trl.txt')
d_label_eval,l_eval,attack_id_eval = split_genSpoof(dir_meta = '/home/crypto/protocols/ASVspoof2019.LA.cm.eval.trl.txt')

# Train Loader

train_set = Dataset_ASVspoof2019_LA(list_IDs = utt_train_path,labels = d_label_trn,base_dir = '/home/crypto/database/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True)

# Dev Loader

dev_set = Dataset_ASVspoof2019_LA(list_IDs = l_dev, labels = d_label_dev, base_dir = '/home/crypto/database/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac/')
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

private_dev_loader = []
for data, target in dev_loader:
    private_dev_loader.append((
        data.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="snn"),
        target.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="snn")
    ))
    
private_dev_loader_2 = private_dev_loader.copy()

# Test Loader

test_set = Dataset_ASVspoof2019_LA(list_IDs = l_eval, labels = d_label_eval, base_dir = '/home/crypto/database/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

private_test_loader = []
for data, target in test_loader:
    private_test_loader.append((
        data.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="snn"),
        target.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="snn")
    ))

private_test_loader_2 = private_test_loader.copy()

#Feed Forward Neural Network specification

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2970, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x,y=None):
        x=x.view(-1,2970)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = Net()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Training function

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_total = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)
        num_total += batch_size
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1)
        loss = criterion(output, target.float())
        loss.backward()
        total_loss += (loss.item() * batch_size)
        optimizer.step()
    total_loss /= num_total
    return total_loss

# Plaintext evaluation

def test(args, model, test_loader, save_path,utt_id_eval,attack_id_eval):
    start_time = time.time()
    model.eval()
    fname_list = []
    key_list = []
    sys_id_list = []
    score_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            batch_score = output.data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id_eval)
            key_list.extend(
              ['human' if key == 1 else 'spoof' for key in list(target)])
            sys_id_list.extend(attack_id_eval)
            score_list.extend(batch_score.tolist())
        
    print(time.time()-start_time)
    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            
    print('Result saved to {}'.format(save_path))

best_loss = 1000.0

# Launching the training

for epoch in range(1, args.epochs + 1):
    train_loss = train(args, model, train_loader, optimizer, epoch)
    epoch_loss = train(args, model, dev_loader, optimizer, epoch)
    print('Train Epoch: {} \tLoss: {:.8f}'.format(epoch, epoch_loss))
    if epoch_loss < best_loss:
        torch.save(model.state_dict(), '/home/chouchan/code/DNN_model.pth')
        best_loss =epoch_loss

model.load_state_dict(torch.load('/home/chouchan/code/DNN_model.pth'))
model_smpc = model.copy().fix_precision()
model_smpc_enc = model.copy().fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="snn")

# Plaintext evaluation with plaintext model

test(args, model, dev_loader, '/home/chouchan/code/results_dev.txt', l_dev, attack_id_dev)
test(args, model, test_loader, '/home/chouchan/code/results_eval.txt', l_eval, attack_id_eval)

# Secure Evaluation

def test(args, model, test_loader, save_path,utt_id_eval,attack_id_eval):
    start_time_test = time.time()
    model.eval()
    fname_list = []
    key_list = []
    sys_id_list = []
    score_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.view(-1).to(device)
            output = model(data)
            batch_score = (output.copy().get()).view(-1)
            batch_target = (target.copy().get())
            # add outputs
            fname_list.extend(utt_id_eval)
            key_list.extend(
              ['human' if key == 1000 else 'spoof' for key in list(batch_target)])
            sys_id_list.extend(attack_id_eval)
            score_list.extend(batch_score.tolist())

    print(time.time()-start_time_test)
    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))

    print('Result saved to {}'.format(save_path))

# Secure evaluation with non encrypted model

test(args, model_smpc, private_dev_loader, '/home/chouchan/code/results_secure_dev.txt', l_dev, attack_id_dev)
test(args, model_smpc, private_test_loader, '/home/chouchan/code/results_secure_eval.txt', l_eval, attack_id_eval)

# Secure evaluation with encrypted model

test(args, model_smpc_enc, private_dev_loader_2, '/home/chouchan/code/results_secure_dev_encmod.txt', l_dev, attack_id_dev)
test(args, model_smpc_enc, private_test_loader_2, '/home/chouchan/code/results_secure_eval_encmod.txt', l_eval, attack_id_eval)
