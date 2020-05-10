from LazyML.DataLoader import GetData
from LazyML.LazyModel import LazyClassifier
from sklearn.model_selection import train_test_split

data_set = GetData('mnist').load_data()

dt_y = data_set['train'].label
dt_x = data_set['train'].drop('label', axis=1)
x_train, x_valid, y_train, y_valid = train_test_split(
    dt_x,
    dt_y,
    test_size=0.20,
    random_state=5566
    )


clf = LazyClassifier(x_train, y_train, x_valid, y_valid, data_set['test'])

model_dict = clf.hyper_dict

for k, v in model_dict.items():
    print()
    print(k)
    print('-' * 40)
    print()
    clf.fit(model_name=k, hyper_space=v['space'], hyper_objective=v['objective'], max_eval=2)
    print()

