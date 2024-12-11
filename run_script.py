import os
import sys

import src.agent.disc_ancnes
from src.agent import *

if __name__ == "__main__":
	# env_vars = os.environ
	# for key, value in env_vars.items():
	# 	print(f"{key}={value}")
	# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
	src.agent.disc_ancnes.main()
