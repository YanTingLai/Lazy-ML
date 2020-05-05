# Lazy-ML
* 內建多種分類器，透過grid search或hyperopt調參

# 準備資料

https://github.com/zalandoresearch/fashion-mnist

參照fashion-mnist下載資料方式

!git clone git@github.com:zalandoresearch/fashion-mnist.git <br>
`import mnist_reader` <br>
`X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')`<br>
`X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')`<br>

