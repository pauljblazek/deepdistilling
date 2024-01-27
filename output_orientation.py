import numpy as np
import random

def ENN_orientation(I):

	D = np.zeros((28, 28))
	for i in range(28):
		for j in range(28):
			col_sum = np.sum(I[:, i])
			row_sum = np.sum(I[j, :])
			if col_sum > row_sum:
				D[i,j] = 1
			elif col_sum < row_sum:
				D[i,j] = -1
			

	S1 = np.zeros(28)
	for i in range(28):
		row_sum = np.sum(D[i, :])
		offrow_sum = (np.sum(D) - np.sum(D[i, :]))
		if offrow_sum < -29:
			S1[i] = 1
		elif offrow_sum > -27:
			S1[i] = -1
		elif offrow_sum == -27:
			if np.all(D[i, :]==1):
				S1[i] = 1
			elif not np.all(D[i, :]==1):
				S1[i] = -1
		elif offrow_sum == -28:
			if row_sum > 0:
				S1[i] = 1
			elif row_sum < 0:
				S1[i] = -1
		elif offrow_sum == -29:
			if not np.all(D[i, :]==-1):
				S1[i] = 1
			elif np.all(D[i, :]==-1):
				S1[i] = -1
		
	S2 = np.zeros(28)
	for i in range(28):
		offcol_sum = (np.sum(D) - np.sum(D[:, i]))
		col_sum = np.sum(D[:, i])
		if offcol_sum < 27:
			S2[i] = -1
		elif offcol_sum > 29:
			S2[i] = 1
		elif offcol_sum == 29:
			if np.all(D[:, i]==1):
				S2[i] = -1
			elif not np.all(D[:, i]==1):
				S2[i] = 1
		elif offcol_sum == 28:
			if col_sum > 0:
				S2[i] = -1
			elif col_sum < 0:
				S2[i] = 1
		elif offcol_sum == 27:
			if not np.all(D[:, i]==-1):
				S2[i] = -1
			elif np.all(D[:, i]==-1):
				S2[i] = 1
		
	C1 = 0.206*np.sum(S1) - 0.206*np.sum(S2)
	C2 = 0.206*np.sum(S2) - 0.206*np.sum(S1)
	C = [C1, C2]
	results = np.where(C==np.max(C))[0]
	return random.choice(results)