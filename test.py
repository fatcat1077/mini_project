import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#############################
# 1. 原始公用函式（維持不變）
#############################

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    # 為了對齊，根據 train loss 與 dev loss 的比例抽取 dev_loss
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            # 只取前 40 states + 幾個 tested_positive 相關特徵
            feats = list(range(40)) + [40,42,43,57,58,60,61, 75,76,78,79]

        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader

#############################
# 2. 定義模型 (NeuralNet)
#############################
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        # 也可自行加 L1 或 L2 regularization
        return self.criterion(pred, target)

#############################
# 3. 定義訓練與驗證流程
#############################
def train_one_epoch(tr_set, model, optimizer, device, loss_record):
    model.train()
    for x, y in tr_set:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse_loss = model.cal_loss(pred, y)
        mse_loss.backward()
        optimizer.step()
        loss_record['train'].append(mse_loss.detach().cpu().item())

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss

def train_and_validate(tr_set, dv_set, model, config, device):
    ''' 
    以 config 中設定的 optimizer 與參數訓練 model，
    回傳最終的 dev_loss 與整個 train/dev loss 紀錄(可用於後續畫圖)。
    '''
    n_epochs = config['n_epochs']
    # 根據 config 建立 optimizer (以 Adam 為例，也可用 SGD、RMSprop 等)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    # 也可根據 config['optimizer'] 動態選擇不同優化器

    best_dev_loss = float('inf')
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        train_one_epoch(tr_set, model, optimizer, device, loss_record)
        dev_loss = dev(dv_set, model, device)
        loss_record['dev'].append(dev_loss)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        if early_stop_cnt > config['early_stop']:
            # early stop
            break

    return best_dev_loss, loss_record

#############################
# 4. 測試 (test) 與 儲存預測 (save_pred)
#############################
def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


#############################
# 5. 主程式：超參數搜尋
#############################
if __name__ == '__main__':
    # 固定隨機種子
    myseed = 42069
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    # 準備路徑
    tr_path = 'covid.train.csv'
    tt_path = 'covid.test.csv'
    os.makedirs('models', exist_ok=True)

    device = get_device()
    target_only = True  # 只用部分特徵

    # 以下定義想要嘗試的超參數組合
    learning_rates = [1e-3, 5e-4, 1e-4]
    weight_decays = [1e-5, 1e-6]
    batch_sizes   = [256, 500]

    # 其他設定(可視情況更動)
    n_epochs = 1000
    early_stop = 200

    # 用來記錄「最佳」結果
    best_loss = float('inf')
    best_config = None
    best_model_state = None
    best_loss_record = None

    # 先不建立 dataloader，因為我們要隨著 batch_size 改變而重建
    for lr in learning_rates:
        for wd in weight_decays:
            for bs in batch_sizes:
                print(f'==== Try (lr={lr}, weight_decay={wd}, batch_size={bs}) ====')

                # 重新建立 dataloader（根據 batch_size）
                tr_set = prep_dataloader(tr_path, 'train', bs, target_only=target_only)
                dv_set = prep_dataloader(tr_path, 'dev', bs, target_only=target_only)

                # 建立模型並搬移到 device
                model = NeuralNet(tr_set.dataset.dim).to(device)

                # 設定當前超參數
                config = {
                    'n_epochs': n_epochs,
                    'learning_rate': lr,
                    'weight_decay': wd,
                    'early_stop': early_stop
                }

                # 進行訓練 (並回傳最終 dev_loss)
                dev_loss, loss_record = train_and_validate(tr_set, dv_set, model, config, device)
                print(f'  --> dev_loss = {dev_loss:.4f}')

                # 若表現比較好，更新紀錄
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_config = (lr, wd, bs)
                    # 用 state_dict() 把目前模型的權重存起來
                    best_model_state = model.state_dict()
                    best_loss_record = loss_record
    
    print('=====================================')
    print(f'Best loss on dev set = {best_loss:.4f}')
    print(f'Best config: lr={best_config[0]}, wd={best_config[1]}, batch_size={best_config[2]}')
    print('=====================================')

    # 用「最佳超參數」重新建 dataloader 與模型，然後載入最佳權重
    best_lr, best_wd, best_bs = best_config
    tr_set = prep_dataloader(tr_path, 'train', best_bs, target_only=target_only)
    dv_set = prep_dataloader(tr_path, 'dev', best_bs, target_only=target_only)
    tt_set = prep_dataloader(tt_path, 'test', best_bs, target_only=target_only)

    best_model = NeuralNet(tr_set.dataset.dim).to(device)
    best_model.load_state_dict(best_model_state)  # 載入最佳權重

    # 你也可以把最佳 model_state 存到檔案，下次就不用重跑
    torch.save(best_model_state, 'models/best_model.pth')

    # 看看最佳模型在 dev_set 的學習曲線
    plot_learning_curve(best_loss_record, title='Best Model')

    # 看看在 dev_set 的預測表現 (散點圖)
    plot_pred(dv_set, best_model, device)

    # 最終在測試集上做預測
    preds = test(tt_set, best_model, device)
    save_pred(preds, 'pred.csv')  # 儲存最終預測結果
