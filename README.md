# Lazy-ML
* 內建多種分類器，透過hyperopt調整參數。

# 使用方式
* 將整理好的資料集丟給LazyClassifier
* 參考example
`from LazyML.LazyModel import LazyClassifier` <br>
`clf = LazyClassifier(x_train, y_train, x_valid, y_valid, x_test)`<br>
`model_dict = clf.hyper_dict`<br>
