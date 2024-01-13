import numpy as np
import random

def ENN_maxsat(I, n, m):

	D1 = np.zeros(n)
	for i in range(n):
		if np.any(I[2:, i]!=0):
			D1[i] = -1
		else:
			D1[i] = 1

	D2 = np.zeros(n)
	for i in range(n):
		if np.any(I[2:, i]!=0):
			D2[i] = 1
		else:
			D2[i] = -1

	D3 = np.zeros(n)
	for i in range(n):
		col_mean = np.mean(I[2:, i])
		if I[1, i] + col_mean - I[0, i] > 0:
			D3[i] = 1
		else:
			D3[i] = -1
		
	D4 = np.zeros(n)
	for i in range(n):
		col_mean = np.mean(I[2:, i])
		if I[0, i] + col_mean - I[1, i] > 0:
			D4[i] = 1
		else:
			D4[i] = -1
		
	D5 = np.zeros(n)
	for i in range(n):
		if I[0, i]:
			D5[i] = 1
		elif (not I[1, 0]) or I[0, i]:
			D5[i] = -1
		
	D6 = np.zeros(n)
	for i in range(n):
		if I[1, i]:
			D6[i] = 1
		elif (not I[0, 0]) or I[1, i]:
			D6[i] = -1
		

	S1 = np.zeros(n)
	for i in range(n):
		S1[i] = (D6[i]>0 and D1[i]>0)
		
	S2 = np.zeros(n)
	for i in range(n):
		S2[i] = (D2[i]>0 and D3[i]>0)
		
	S3 = np.zeros(n)
	for i in range(n):
		S3[i] = (D5[i]>0 and D1[i]>0)
		
	S4 = np.zeros(n)
	for i in range(n):
		S4[i] = (D2[i]>0 and D4[i]>0)
		
	C1 = 10.0*np.sum(S3) + 2.298*np.sum(S4) - 2.298*np.sum(S2) - 10.0*np.sum(S1)
	C2 = 10.0*np.sum(S1) + 2.298*np.sum(S2) - 2.298*np.sum(S4) - 10.0*np.sum(S3)
	C = [C1, C2]

	return np.exp(C)/np.sum(np.exp(C))