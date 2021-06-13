import pickle


with open("Top1000DesignsRandomSearch", "rb") as file:
    data = pickle.load(file)

print(len(data))

count = 0
for j in range(1000):
    for i in range(6):
        for ii in range(4):
                if data[j][str(i)+str(ii)+"0"]:  # True means the reuploading is on
                    count +=1

print(count / (24*1000))
