import json
from glob import glob

import torch
from lstm_genre_classification import AudioDataset, AudioClassifier
from prepare_data import get_id_from_path
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", default="/media/ml/data_ml/fma_metadata/")
    parser.add_argument("--mp3_path", default="/media/ml/data_ml/fma_small/")

    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    mp3_path = Path(args.mp3_path)

    batch_size = 32
    epochs = 64

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

    reconstruction_weights = [100.0, 10.0, 5.0, 2.0, 1.0, 0.1, 0.01, 0.0]

    model_paths = [
        list(glob("../models/model_%s*" % a))[0] for a in reconstruction_weights
    ]

    models = [AudioClassifier.load_from_checkpoint(a).cuda() for a in model_paths]

    accuracies = []

    for model in tqdm(models):
        correct = 0
        model.eval()

        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x = x.cuda()
                y = y.cuda()
                y_pred, _ = model(x)
                y_pred = F.softmax(y_pred, dim=-1)
                _, y_pred = torch.max(y_pred, dim=-1)
                correct += (y_pred == y).sum().item()

        accuracies.append(correct / len(test_data))

    data = {"reconstruction_weights": reconstruction_weights, "accuracies": accuracies}

    json.dump(data, open("../eval_data/_test_accuracy.json", "w"), indent=4)
