import pandas as pd
import numpy as np
from functools import partial
from data import train_set, test_set
from Kernels import mismatch_kernel
from model import KSVM

if __name__ == "__main__":
	X_path = 'Training/Xtr0.csv'
	Y_path = 'Training/Ytr0.csv'
	trainset, label = train_set(X_path,Y_path, seq = ['A', 'C', 'G', 'T'])

	kernel = partial(mismatch_kernel, length = 5, mismatch = 1, λ = .5, norm = True)
	model = KSVM(kernel, reg = 1e-4)
	model.fit(trainset, label)
	test_path = 'Test/Xte0.csv'
	testset = test_set(test_path, seq = ['A', 'C', 'G', 'T'])
	test_label0 = model.predict(testset)

	X_path = 'Training/Xtr1.csv'
	Y_path = 'Training/Ytr1.csv'
	trainset, label = train_set(X_path,Y_path, seq = ['A', 'C', 'G', 'T'])

	kernel = partial(mismatch_kernel, length = 5, mismatch = 1, λ = .5, norm = True)
	model = KSVM(kernel, reg = 1e-3)
	model.fit(trainset, label)
	test_path = 'Test/Xte1.csv'
	testset = test_set(test_path, seq = ['A', 'C', 'G', 'T'])
	test_label1 = model.predict(testset)

	X_path = 'Training/Xtr2.csv'
	Y_path = 'Training/Ytr2.csv'
	trainset, label = train_set(X_path,Y_path, seq = ['A', 'C', 'G', 'T'])

	kernel = partial(mismatch_kernel, length = 5, mismatch = 1, λ = .5, norm = True)
	model = KSVM(kernel, reg = 5e-4)
	model.fit(trainset, label)
	test_path = 'Test/Xte2.csv'
	testset = test_set(test_path, seq = ['A', 'C', 'G', 'T'])
	test_label2 = model.predict(testset)

	test_label = np.vstack([np.array([i for i in range(3000)]), np.hstack([test_label0, test_label1, test_label2])]).T.astype('int')
	test_label = pd.DataFrame(test_label, columns = ['Id', 'Bound'])
	test_label.to_csv('Yte.csv', index = False)
