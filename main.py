import torch
from dataloader import VertebraeDataset
import argparse
import loguru
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
import os
from tqdm import tqdm
from model import BMDNeXt
from loss import weighted_mse_loss
from lds import ms_alds

from sklearn.model_selection import GroupKFold

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--model_name', type=str, default='BMDNeXt')
parser.add_argument('--seed', type=int, default=4396)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--start_fds_epoch', type=int, default=4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)

model_dir = os.path.join('./record', args.model_name)
os.makedirs(model_dir, exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)

logger = loguru.logger
log_file = os.path.join(model_dir, f'train_{args.model_name}.log')
logger.add(log_file, rotation='10 MB')

def train(epoch):
    model.train()
    train_loss = 0
    global optimizer
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.start_epoch + args.epoch - 1}', unit='batch') as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            model.set_train(True)

            data, target = data.to(device), target.to(device)

            weights = ms_alds(target)[2]

            output, training_features, all_features = model(data, target, epoch)

            # visualize_input_and_features(data,all_features,1,64)
            
            loss = weighted_mse_loss(output.flatten(), target.flatten(), weights=weights)
            loss.backward()
            optimizer.step()

            if epoch >= args.start_fds_epoch:
                model.AFDS.update_last_epoch_stats(epoch)
                model.AFDS.update_running_stats(training_features, target, epoch)
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)
            
def validate(epoch,fold=None):
    global min_mae
    model.set_train(False)
    model.eval()
    targets = np.array([])
    predictions = np.array([])
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ , _= model(data, target, epoch)
            targets = np.append(targets, target.detach().cpu().numpy())
            predictions = np.append(predictions, output.detach().cpu().numpy())

    mae = mean_absolute_error(targets, predictions)
    targets.reshape(-1, 1)
    predictions.reshape(-1, 1)

    pcc, _ = stats.pearsonr(targets, predictions)
    
    return mae, pcc

def cross_validate(folds=5):
    global min_mae, optimizer, train_loader, val_loader, model
    
    dataset = VertebraeDataset(args.datapath, mode='all')
    
    # Get patient ID
    patient_id = dataset.id
    
    group_kfold = GroupKFold(n_splits=folds)
    
    maes = []
    pccs = []
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X=range(len(dataset)), groups=patient_id)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        # Reinitialize the model and optimizer
        model = BMDNeXt(start_fds_epoch=args.start_fds_epoch).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)

        min_mae = float('inf')

        logger.info(f"Starting fold {fold+1}/{folds}")

        for epoch in range(args.epoch):
            train(epoch)
            mae, pcc = validate(epoch, fold)
            scheduler.step()
            logger.info(f'Fold {fold+1} Epoch {epoch}/{args.epoch - 1} - MAE: {mae:.4f} PCC: {pcc:.4f}')

        maes.append(mae)
        pccs.append(pcc)

    average_mae = np.mean(maes)
    average_pcc = np.mean(pccs)
    logger.info(f'5-Fold CV Average MAE: {average_mae:.4f}, Average PCC: {average_pcc:.4f}')

    return maes, pccs

def main():
    cross_validate(5)

if __name__ == '__main__':
    main()