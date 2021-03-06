import numpy as np

r = np.array( [[3,4,5],[9,8,7]])

s = np.array( [[1,2],[2,3]])
u = np.array([[99,88]])
s = np.concatenate( (s,u.T) , axis =1)


t = np.concatenate((r, s),axis=0)

print(t)
# like r
data = np.loadtxt("train.nmv.txt")

prelimData = np.genfromtxt("prelim-nmv-noclass.txt")
prelimData = [i[:-1] for i in prelimData]    
prelimData = np.array(prelimData)

prelimClasses = np.loadtxt("prelim-class.txt")
prelimClasses = np.array([prelimClasses])

fullPrelim = np.concatenate( (prelimData, prelimClasses.T), axis = 1)

AllData = np.concatenate((data, fullPrelim),axis=0)

'''
#These prints show the data is combining correctly
print(len(AllData))
print(len(fullPrelim), len(fullPrelim[0]))
print(len(AllData[0]))
'''


with open("AllData.txt","w") as f:
   for item in AllData:
		line = " ".join([str(x) for x in item])
		line += "\n"
		f.write(line)
