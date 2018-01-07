# EEG-Alcoholic
Experimenting with different machine learning techniques to determine if a subject is an alcoholic from EEG data.

Data Source: http://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/

Download eeg_full.tar, and unpack it, then unpack all .gz files in all of it's subdirectories.
Then run the generate-TFRecords notebook to convert data the TFRecords format (this takes a while!). You can then run the run_training notebook to train a convolution neural net to predict if the subject is or is not an alcoholic from looking at a single trial of the experiment.
