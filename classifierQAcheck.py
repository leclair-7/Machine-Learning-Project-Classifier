# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:35:48 2016

@author: leh140230
"""

import numpy as np

ExtraTreeRes = np.loadtxt('Result_ExtraTrees_prelim.txt')
linearSVCRes = np.loadtxt('Result_LinearSVC_prelim_.txt')
LogistRegRes = np.loadtxt('Result_LogisticRegres_prelim_.txt')

#for some reason adaBoost-Bayes sucks balls ~50%, this didn't seem to be the case 10 minutes ago
#ada_bayesRes = np.loadtxt('BayesAdaBoost_prelim_Result.txt')


assert len(ExtraTreeRes) == len(linearSVCRes) and len(LogistRegRes) == len(linearSVCRes)

signal = [0] * len(linearSVCRes)

num0total =0
num1total =0
for i in range(len(linearSVCRes) ):
	if ExtraTreeRes[i] == 1:
		signal[i] += 1
		num1total += 1
	else:
		num0total += 1
	if linearSVCRes[i] == 1:
		signal[i] += 1
		num1total += 1
	else:
		num0total += 1
	if LogistRegRes[i] == 1:
		signal[i] += 1
		num1total += 1
	else:
		num0total += 1

big_picture = [0,0,0,0]
for i in signal:
	if i==0:
		big_picture[0] += 1
	elif i==1:
		big_picture[1] += 1
	elif i==2:
		big_picture[2] += 1
	elif i==3:
		big_picture[3] += 1

prelimClasses = np.loadtxt("prelim-class.txt")

preL_info = [0,0]
for k in prelimClasses:
	if k ==0:
		preL_info[0] += 1
	elif k==1:
		preL_info[1] += 1

print("Below are number of 1 classifications from the 3 models:")
print("Also, number of 0's and 1's in the preliminary classes :")		
print( big_picture, ", classifiers [0,1,2,3] ; prelimClasses [|0|,|1|]--> ",preL_info)
print("It seems as though it is possible to get a good run from this info")
print("but the schemes below are crap")
h = []
for i in range(len(prelimClasses)):
    if ExtraTreeRes[i] + linearSVCRes[i] + LogistRegRes[i] ==2:
        h.append(1)
    elif ExtraTreeRes[i] + linearSVCRes[i] + LogistRegRes[i] ==3:
        h.append(1)
    else:
        h.append(0)

right =0
wrong =0
for i in range(len(h) ):
	if h[i] ==  prelimClasses[i]:
		right += 1
	else:
		wrong += 1
right = float(right)
wrong = float(wrong)
print("2 or 3 the yes, score is: ", (right/(right + wrong)))

#-------------------------------

h = []
for i in range(len(prelimClasses)):
    if ExtraTreeRes[i] + linearSVCRes[i] + LogistRegRes[i] !=0:
        h.append(1)    
    else:
        h.append(0)

right =0
wrong =0
for i in range(len(h) ):
	if h[i] ==  prelimClasses[i]:
		right += 1
	else:
		wrong += 1
right = float(right)
wrong = float(wrong)
print("Any classified 1 then yes score is: ", (right/(right + wrong)))

#----------------------------------
#----------------------------------

import random
h = []
for i in range(len(prelimClasses)):
    if ExtraTreeRes[i] + linearSVCRes[i] + LogistRegRes[i]  == 3:
        h.append(1)    
    elif ExtraTreeRes[i] + linearSVCRes[i] + LogistRegRes[i]  == 0:
        h.append(0)
    else:
        h.append(random.randint(0,1) )       

right =0
wrong =0
for i in range(len(h) ):
	if h[i] ==  prelimClasses[i]:
		right += 1
	else:
		wrong += 1
right = float(right)
wrong = float(wrong)
print("3/3 on classified then yes score is: ", (right/(right + wrong)))

#----------------------------------
