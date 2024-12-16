import matplotlib.pyplot as plt
import os
import csv, json
import numpy as np

palette = plt.get_cmap("Set1")


def show_multi_line_and_area(x, y, fig_name, image_file_name, line_labels, y_label, x_label="Iteration Step",
                             legend=True):
	plt.figure(figsize=(8, 6))
	for i, data, name, x_ in zip(range(len(line_labels)), y, line_labels, x):
		color = palette(i)
		data = np.array(data)
		if len(data.shape) == 1:
			avg = data
		else:
			avg = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			up = avg + std
			down = avg - std
			down = np.maximum(down, 0)
			plt.fill_between(x_, down, up, color=color, alpha=0.2, )
		plt.plot(x_, avg, marker=None, linestyle='-', label=name, color=color)
	
	if legend:
		plt.legend()
	plt.title(fig_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(os.path.join(f'./{image_file_name}.png'))


def read_csv_to_dict(file_name):
	with open(file_name, 'r') as f:
		reader = csv.reader(f)
		headers = next(reader)
		rows = tuple(reader)
		columns = zip(*rows)
		dic = {header: column for header, column in zip(headers, columns)}
	return dic


if __name__ == "__main__":
	lines = []
	
	# res_path = "/res/"
	# for i in os.listdir(res_path):
	# 	if i[-1] == '8':
	# 		continue
	# 	if i[-3] == 'dqn':
	# 		continue
	# 	dic = read_csv_to_dict(os.path.join(res_path, i, "progress.csv"))
	# 	name = i.split("_")[2].upper()
	# 	l = [(0.0 if s == '' else float(s)) for s in dic['rollout/ep_rew_mean']]
	# 	lines.append((name, l))
	
	for i in os.listdir('/results/A-NCNES'):
		if i.split("_")[0] == "10-Freeway":
			continue
		datas = []
		for j in os.listdir(os.path.join('/results/A-NCNES', i)):
			if j == 'sum_result.png': continue
			with open(os.path.join('/results/A-NCNES', i, j, "config_result.json")) as f:
				c = json.load(f)
				data = c['average_test_score']
				datas.append(data)
		name = i.split('_')[-2].upper()
		lines.append((name, datas))
	
	ys = []
	xs = []
	ns = []
	for name, i in lines:
		ns.append(name)
		n = np.array(i)
		xs.append(n)
		n = n.shape[-1]
		ys.append([(i + 1) / n for i in range(n)])
	show_multi_line_and_area(ys, xs, 'Enduro with Elite Protection', 'image_', ns, 'Score', 'Training Process')
	
	print()
