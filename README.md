# EmoContext


```
docker build -t emo .
 nvidia-docker run -it -v "$PWD":/app -p 8888:8888 emo
```

### Feature Vectors:
```
notebooks> gdrive upload x_test 
Uploading x_test
Uploaded 17BXf7v7TzZv8UpzrCZi6TTQqwzKXtg97 at 27.3 MB/s, total 2.6 GB
notebooks> gdrive upload y_test 
Uploading y_test
Uploaded 1dyo6wdJG6C2wJUToSfLlIOCxHzfuCTfH at 116.3 KB/s, total 120.7 KB
```


# Lame notes:

```

get_model_1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Without using class weights
model.fit(x_train,
          y_train,
          validation_data=(np.array(x_test),y_test),
          shuffle=True,
          batch_size=124,
          epochs=20)

('True Positives per class : ', array([ 4143.,  1213.,  1507.,  1664.]))
('False Positives per class : ', array([ 313.,  458.,  215.,  440.]))
('False Negatives per class : ', array([ 784.,  158.,  321.,  163.]))
Class happy : Precision : 0.726, Recall : 0.885, F1 : 0.798
Class sad : Precision : 0.875, Recall : 0.824, F1 : 0.849
Class angry : Precision : 0.791, Recall : 0.911, F1 : 0.847
Ignoring the Others class, Macro Precision : 0.7973, Macro Recall : 0.8733, Macro F1 : 0.8336
Ignoring the Others class, Micro TP : 4384, FP : 1113, FN : 642
Accuracy : 0.8567, Micro Precision : 0.7975, Micro Recall : 0.8723, Micro F1 : 0.8332

# While using class weights

('True Positives per class : ', array([ 4443.,  1055.,  1522.,  1516.]))
('False Positives per class : ', array([ 682.,  298.,  228.,  209.]))
('False Negatives per class : ', array([ 484.,  316.,  306.,  311.]))
Class happy : Precision : 0.780, Recall : 0.770, F1 : 0.775
Class sad : Precision : 0.870, Recall : 0.833, F1 : 0.851
Class angry : Precision : 0.879, Recall : 0.830, F1 : 0.854
Ignoring the Others class, Macro Precision : 0.8428, Macro Recall : 0.8106, Macro F1 : 0.8264
Ignoring the Others class, Micro TP : 4093, FP : 735, FN : 933
Accuracy : 0.8576, Micro Precision : 0.8478, Micro Recall : 0.8144, Micro F1 : 0.8307

# Using model two (regularised):
('True Positives per class : ', array([ 4135.,  1189.,  1626.,  1633.]))
('False Positives per class : ', array([ 289.,  412.,  342.,  327.]))
('False Negatives per class : ', array([ 792.,  182.,  202.,  194.]))
Class happy : Precision : 0.743, Recall : 0.867, F1 : 0.800
Class sad : Precision : 0.826, Recall : 0.889, F1 : 0.857
Class angry : Precision : 0.833, Recall : 0.894, F1 : 0.862
Ignoring the Others class, Macro Precision : 0.8007, Macro Recall : 0.8835, Macro F1 : 0.8401
Ignoring the Others class, Micro TP : 4448, FP : 1081, FN : 578
Accuracy : 0.8624, Micro Precision : 0.8045, Micro Recall : 0.8850, Micro F1 : 0.8428
```