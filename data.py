import numpy as np

def split_file(filename):
	fp = open(filename, 'r')
	p_ = []
	max=0
	t=0
	for line in fp:
		if t > 1000:
			break
		t+=1
		p_line = []
		for w in line.split():
			p_line.append(w)
		if len(p_line) > max:
			max = len(p_line)
		# p_line = np.asarray(p_line, dtype=str)
		# p_line = p_line.reshape(1, p_line.shape[0])
		p_.append(p_line)
		
	for l in p_:
		i = len(l)
		while i<max:
			l.append(0)
			i+=1

	return np.array(p_), max

def pad(data):
	max=0
	for l in data:
		if len(l)>max:
			max = len(l)
	print("Max length %d"%max)
	padded = []
	for line in data:
		l_ = len(line)
		while l_ < max:
			np.vstack([line, np.zeros(50)])
			l_+=1
			# padded.append(np.pad(line, (0, max-line.shape[0]), "constant", constant_values=('0')))
		padded.append(np.array(line))
	return np.array(padded)			
	

def load_train_data():
	premise, max_p = split_file("SNLI/s1.train")
	hypothesis, max_h = split_file("SNLI/s2.train")
	labels, _ = split_file("SNLI/labels.train")
	return premise, max_p, hypothesis, max_h, labels, _