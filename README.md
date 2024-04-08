# A transformer model that predicts the next values of multiple, related, time series

## To run the model
1. Install the required packages listed in `requirements.txt`
2. Run the ``train.py`` script to train the model
   - You may want to adjust the hyperparameters in the script, in particular the number of epochs, depending on your hardware
   - I have not tested the model extensively with different epochs, the predictions are statistically significant with 30,000 epochs and default parameters but this takes a long time to train
   - Stock data is downloaded from Yahoo Finance and cached in the ``data_filename`` file.
   - The model is saved in the ``model_filename`` file.
   - The script will also save the scaler used to normalize the data in the ``scaler_filename`` file.
3. Run the ``test_model.py`` script to test the model
   - This will generate a plot of the model's predictions, run a T-test to determine if the predictions are statistically significant, and print the mean squared error of the model's predictions
4. Run the ``virtual_trading.py`` script to simulate trading based on the model's predictions
   - This will simulate trading in February based on the data from January with default parameters
   - Note: Yahoo Finance data seems to update at EOM for some stocks, so setting the end data to any day in the current month will not work
   

## Additional information

- The docker file will build an image that runs the training script only.
- The model will be deleted after training, if you want to keep it you should move/rename it.
