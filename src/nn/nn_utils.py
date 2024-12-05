import src.nn
import torch


def get_network_class(name):
	try:
		nn_class = getattr(src.nn, name)
	except AttributeError as e:
		print(f'AttributeError: {e}. Check the model class name and the file "src/nn/__init__.py".')
		quit()
	
	return nn_class


def get_device():
	# if torch.cuda.is_available():
	# 	return "cuda"
	return "cpu"


class BaseModule(torch.nn.Module):
	def get_parameters(self):
		# 获取网络参数作为一个扁平化的1D数组
		return torch.cat([param.view(-1) for param in self.parameters()])
	
	def get_parameters_amount(self):
		return sum(param.numel() for param in self.parameters())

	def set_parameters(self, flat_parameters):
		# 将扁平化的1D数组参数设置回网络
		offset = 0
		for param in self.parameters():
			param_size = param.numel()
			param.data.copy_(flat_parameters[offset:offset+param_size].view(param.size()))
			offset += param_size

if __name__ == '__main__':
	print(f"Test: src.nn.nn_utils.")
	strs = "dqn"
	nn_class = get_network_class(strs)
	print(nn_class)
	

	