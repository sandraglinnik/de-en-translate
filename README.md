This project contains 3 models for translation:

- __Naive word-by-word model__ (`naive_model.py`) matches texts in different languages in order to figure out the most
  common translation for each word. Can be used as a simple baseline.
- __Context LSTM__ (`context_lstm.py`) consists of two LSTMs, which are encoder and decoder. Encoder's final state is
  passed as decoder's initial state.
- __Transformer__ (`transformer.py`) represents classical seq-to-seq transformer architecture.

`main.py` file contains examples of training and usage of these models.