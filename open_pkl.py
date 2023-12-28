import pickle

f = open('reports.pkl','rb')
data = pickle.load(f)
print(data)