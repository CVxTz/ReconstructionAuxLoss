#python prepare_data.py --metadata_path "/media/ml/data_ml/fma_metadata/"
#python audio_processing.py --mp3_path "/media/ml/data_ml/fma_small/"

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


python eval_lstm_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/"