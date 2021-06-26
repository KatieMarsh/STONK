# StockPredictor
it will predict stock price(it will deafult to KBANK.BKK stock but if you want you can edit it to do other stock as well).
There are MANY flaws with my edits so use it at your own risk.

It work by using LSTM and stock indicator(well, so much for that xD) to predict the stock price on close.

The trained model I use is trained with data from 1990-01-01(1st of jan 1990) to 2019-01-01(1st of jan 2019). I evaluate the model with data from 2019-01-02 to 2021-06-25.

From my evaluation Evaluate_model_sma_t3.ipynb was the closest to the correct price(which is still pretty far off) but still it provide some proofs that with the right indicator you can inceases the effectiveness of the model.
