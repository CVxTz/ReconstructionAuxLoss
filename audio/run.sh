# Data download:

#wget -c https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
#wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Data preparation

python prepare_data.py --metadata_path "/media/ml/data_ml/fma_metadata/"
python audio_processing.py --mp3_path "/media/ml/data_ml/fma_small/"

# Training

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 100.

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 10.

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 5.

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 2.

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 1.

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 0.1

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 0.01

python lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --reconstruction_weight 0.0

# Evaluation

python eval_lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/"