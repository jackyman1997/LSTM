# LSTM Autoencoder
This practise project attempts to use LSTM Autoencoder to mimic or forcast data. Data I used here is from my own work on [scraping HSI futures minute data](https://github.com/jackyman1997/aastock) from [AAstock](http://www.aastocks.com/tc/). 

# Notes
[rule for the number of hidden nodes](https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm)  
[LSTM pytorch docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)  
[for error `RuntimeError: input.size(-1) must be equal to input_size`](https://discuss.pytorch.org/t/why-do-i-get-input-size-1-must-be-equal-to-input-size-error-in-this-case/120685/13)  
[difference of `.flatten()` and `.view(-1)` in PyTorch](https://discuss.pytorch.org/t/what-is-the-difference-of-flatten-and-view-1-in-pytorch/51790)  

# Disclaimer
This practise project shall never be used for predicting any sort of finanical figures or data, and shall never trust the outcome generated from this project. Basically, this is a practise on how to convert time-series data into pytorch-ready tensors only, the LSTM model training is not the main idea. 