## What's new?

- new classes *FRLN* and *FRLNforClueNER* in models.py, implementing the proposed FRLN model.
- new classes *NERDatasetForFRLN* and *ClueNERDataset* in dataloader.py, modifying the original dataset to meet the need of training the proposed FRLN model.
- *draw.ipynb* in the folder ./logs to collect the loss in the log output and plot the curve.

## Train method 

Open main.py and run to train and test the Bert+BiLSTM+CRF.

Open test_model.py and run to train and test the proposed FRLN model.

Open test_model_on_ClueNER.py and run to train the proposed FRLN model on the ClueNER2020 dataset. The metrics algorithm on the ClueNER2020 has not implemented.

Modify the config.py to customize the training settings.

For more details, go to the github repository in the Reference.

## Reference

https://github.com/tmylla/NER_ZH
