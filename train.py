import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix
import json


from src.dataset import ASTDataset
from src.model import CustomAST
from src.sam import SAM

def train(args):
   
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {DEVICE}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    
    print(f"📥 Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {args.data_path}. Lütfen preprocess.py çalıştırın.")

    data = np.load(args.data_path)
    X_train, y_train, d_train = data['X_train'], data['y_train'], data['device_train']
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    
    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    counts = np.bincount(y_train)
    weights = [1.0/counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    
    dl_kwargs = dict(batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     pin_memory=(DEVICE.type == 'cuda') if args.pin_memory is None else args.pin_memory)

    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True), 
        sampler=sampler,
        **dl_kwargs
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False), 
        shuffle=False,
        **dl_kwargs
    )

    # 3. MODEL VE OPTIMIZER
    print("🧠 Preparing")
    model = CustomAST(num_classes=4).to(DEVICE)

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Disable AMP for P100 (weak float16 support on older GPUs)
    use_amp = False
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        # Enable AMP only for compute capability >= 7.0 (P100 is 6.0)
        use_amp = props.major >= 7
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # 4. EĞİTİM DÖNGÜSÜ
    print("🚀 Train begins")
    best_score = 0.0
    start_epoch = 0

    # Resume support
    if args.resume:
        ckpt_path = args.resume if os.path.isabs(args.resume) else os.path.join(args.checkpoint_dir, args.resume)
        if os.path.exists(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'])
            try:
                optimizer.base_optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception as e:
                print(f"Warning: could not fully restore optimizer state: {e}")
            best_score = ckpt.get('best_score', 0.0)
            start_epoch = ckpt.get('epoch', 0) + 1
            if scaler is not None and 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception:
                    pass

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # --- Forward + SAM steps with optional AMP ---
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                # unscale before SAM step
                scaler.unscale_(optimizer.base_optimizer)
                optimizer.first_step(zero_grad=True)

                # second forward
                with torch.amp.autocast('cuda'):
                    logits2 = model(inputs)
                    loss2 = criterion(logits2, labels)
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer.base_optimizer)
                optimizer.second_step(zero_grad=True)
                # update scaler (we don't call scaler.step because SAM steps internally call optimizer)
                try:
                    scaler.update()
                except Exception:
                    pass
                cur_loss = loss.item()
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                criterion(model(inputs), labels).backward()
                optimizer.second_step(zero_grad=True)
                cur_loss = loss.item()

            running_loss += cur_loss
            progress_bar.set_postfix({'Loss': cur_loss})

        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                
                logits = model(inputs) 
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        
        cm = confusion_matrix(all_labels, all_preds)
        se = np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]) if np.sum(cm[1:, :]) > 0 else 0
        sp = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
        score = (se + sp) / 2

        print(f"Epoch {epoch+1}: Avg Loss={running_loss/len(train_loader):.4f} | Score={score:.4f} (Se={se:.2f}, Sp={sp:.2f})")

        #
        if score > best_score:
            best_score = score
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            # Save rich checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.base_optimizer.state_dict(),
                'best_score': best_score,
                'args': vars(args)
            }
            if scaler is not None:
                try:
                    ckpt['scaler'] = scaler.state_dict()
                except Exception:
                    pass
            torch.save(ckpt, save_path)
            print(f"    --> 💾 Last best Saved ({best_score:.4f}) @ {save_path}")

        # also write a small run config for reproducibility
        try:
            with open(os.path.join(args.checkpoint_dir, 'run_config.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)
        except Exception:
            pass

    print(f"\n🏆 Best Score: {best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AST model with SAM for ICBHI")
    parser.add_argument("--data_path", type=str, default="./icbhi_ast_16k_8s_metadata.npz", help="Path to processed .npz file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--pin_memory", type=lambda x: x.lower() in ['true','1','yes'], default=None,
                        help="Pin memory for DataLoader (true/false). Defaults to True on CUDA)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint filename to resume from (in checkpoint_dir) or absolute path")
    
    args = parser.parse_args()
    train(args)