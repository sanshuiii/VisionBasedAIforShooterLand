from custom_model import ACCNNModel
import pickle
import time

model = ACCNNModel(observation_space = [128,128,3], action_space = 5)

while True:
    with open("/root/model/model.pt", 'rb') as f:
        model.set_weights(pickle.load(f))

    filename = "/root/data2/0.pkl"
    df = open(filename,'rb')
    data = pickle.load(df)

    data1 = []
    for x in data:
        action, p, v = model.forward([x[0]])
        print(action, p, v)
        temp = [x[0], action[0], x[2], p[0], v[0], x[5]]
        data1.append(temp)

    with open('/root/data/0.pkl', 'wb') as f:
        pickle.dump(data1, f)
    time.sleep(20)

