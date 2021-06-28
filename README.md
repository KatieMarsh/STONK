# StockPredictor
it will predict stock price(it will deafult to KBANK.BKK stock but if you want you can edit it to do other stock as well).
There are MANY flaws with my edits so use it at your own risk.

Since this model is a time serie model you have to train for each and every stocks that you want to put in.

It work by using LSTM sotck predictor model [(Original code)](https://github.com/datawiz-thailand/tutorials) and stock indicator(well, so much for that xD) to predict the stock price on close. (but do be ware if you enter some particular date as input it will creates a bug where true_price is consistently offset)

The trained model I use is trained with data from 1990-01-01(1st of jan 1990) to 2019-01-01(1st of jan 2019). I evaluate the model with data from 2019-01-02 to 2021-06-25.

From my evaluation model with SMA and T3 indicators was the closest to the correct price(which is still pretty far off) but still it provide some proofs that with the right indicator you can inceases the effectiveness of the model.


The web application I will be deploying will contain a different model(but still a SMA+T3 model) to the one I used in evaluation notebook.
