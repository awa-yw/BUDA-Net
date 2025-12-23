jupyter nbconvert \
  --to notebook \
  --execute base.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output train_out.ipynb
