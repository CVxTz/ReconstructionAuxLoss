import json
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from audio_processing import random_crop
from prepare_data import get_id_from_path
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=512):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx][0]
        label = self.data[idx][1]

        array = np.load(npy_path)

        array = random_crop(array, crop_size=self.max_len)

        tokens = torch.tensor(array, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return tokens, label


class AudioClassifier(pl.LightningModule):
    def __init__(self, classes=8, input_size=128, reconstruction_weight=0.1, p=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.reconstruction_weight = reconstruction_weight
        self.input_size = input_size
        self.p = p

        self.do = torch.nn.Dropout(p=self.p)

        self.lstm1 = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=2 * self.input_size,
            hidden_size=self.input_size,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = torch.nn.Linear(self.input_size * 2, self.input_size)
        self.fy = torch.nn.Linear(self.input_size, classes)

        self.fc2 = torch.nn.Linear(self.input_size * 2, input_size)

    def forward(self, x):
        x = self.do(x)

        x, _ = self.lstm1(x)
        x_seq, _ = self.lstm2(x)

        x, _ = torch.max(self.do(x_seq), dim=1)

        x = F.relu(self.do(self.fc1(x)))
        y_hat = self.fy(x)

        x_reconstruction = torch.clamp(self.fc2(self.do(x_seq)), -1.0, 1.0)

        return y_hat, x_reconstruction

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat, x_reconstruction = self(x)

        loss_y = F.cross_entropy(y_hat, y)
        loss_x = F.l1_loss(x, x_reconstruction)

        return loss_y + self.reconstruction_weight * loss_x

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat, x_reconstruction = self(x)

        loss_y = F.cross_entropy(y_hat, y)
        loss_x = F.l1_loss(x, x_reconstruction)

        loss = loss_y + self.reconstruction_weight * loss_x

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_loss_y", loss_y)
        self.log("valid_loss_x", loss_x)
        self.log("valid_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat, x_reconstruction = self(x)

        loss_y = F.cross_entropy(y_hat, y)
        loss_x = F.l1_loss(x, x_reconstruction)

        loss = loss_y + self.reconstruction_weight * loss_x

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_loss_y", loss_y)
        self.log("test_loss_x", loss_x)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.97
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path")
    parser.add_argument("--mp3_path")
    parser.add_argument("--reconstruction_weight", type=float)

    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    mp3_path = Path(args.mp3_path)

    batch_size = 32
    epochs = 256
    reconstruction_weight = args.reconstruction_weight

    CLASS_MAPPING = json.load(open(metadata_path / "mapping.json"))
    id_to_genres = json.load(open(metadata_path / "tracks_genre.json"))
    id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    files = sorted(list(glob(str(mp3_path / "*/*.npy"))))

    labels = [CLASS_MAPPING[id_to_genres[int(get_id_from_path(x))]] for x in files]
    print(len(labels))

    samples = list(zip(files, labels))

    _train, test = train_test_split(
        samples, test_size=0.2, random_state=1337, stratify=[a[1] for a in samples]
    )

    train, val = train_test_split(
        _train, test_size=0.1, random_state=1337, stratify=[a[1] for a in _train]
    )

    train_data = AudioDataset(train)
    test_data = AudioDataset(test)
    val_data = AudioDataset(val)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    model = AudioClassifier(reconstruction_weight=reconstruction_weight)

    logger = TensorBoardLogger(
        save_dir="../",
        version="Lambda=%s" % reconstruction_weight,
        name="lightning_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc",
        mode="max",
        filepath="../models/",
        prefix="model_%s" % reconstruction_weight,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[DecayLearningRate()],
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)
