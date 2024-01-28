import os
import matplotlib.pyplot as plt
from PIL import ImageDraw

import torchvision.transforms as transforms

class logger():
	# Initialize class
	def __init__(self, dataset='', model='', is_adapt='', ave_metrics='', size='', max_G=-1, stamp=0, params=None) -> None:
		self.dataset = dataset
		self.model = model
		self.is_adapt = is_adapt
		self.ave_metrics = ave_metrics
		self.size = size
		self.max_G = max_G
		self.stamp = stamp
		self.params = params
		self.path = f"logs/log_{self.stamp+1}/{self.model}/{'adaptive' if self.is_adapt else 'trad'}/{self.dataset}"

	def set_new_info(self, dataset='', model='', is_adapt='', ave_metrics='', size=''):
		self.dataset = self.dataset if dataset == '' else dataset
		self.model = self.model if model == '' else model
		self.is_adapt = self.is_adapt if is_adapt == '' else is_adapt
		self.ave_metrics = self.ave_metrics if ave_metrics == '' else ave_metrics
		self.size = self.size if size == '' else size
		self.path = f"logs/log_{self.stamp+1}/{self.model}/{'adaptive' if self.is_adapt else 'trad'}/{self.dataset}"

	def log_parameters(self, parameters):
		log_dir = f"logs/log_{self.stamp+1}/"
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		f = open(log_dir + f'parameters.txt', "a")
		summary = open(f"logs/log_{self.stamp+1}/" + f'summary.txt', "a")
		summary.write(f"Log {self.stamp+1}"+ '\n\n')
		summary.close()
		for p in range(len(parameters)):
			f.write(f"{parameters[p][0]}: {parameters[p][1]}"+ '\n')
		f.close()

	def log_metric(self, metrics, img_id, data, run_num=None, attack_name=''):
		summary = open(f"logs/log_{self.stamp+1}/" + f'summary.txt', "a")
		# summary.write(f"Log {self.stamp+1}"+ '\n\n')
		summary.write(f"Attack: {attack_name} on {self.model}" + '\n')
		summary.write(f"Img: {self.dataset} - {img_id}"+ '\n')
		if self.max_G > 0:
			summary.write(f"{'-SUCCESS-' if self.max_G > data[2] else '-FAIL-'}"+ '\n')
		for i in range(len(metrics)):
			# check whether directory already exists
			if not os.path.exists(self.path + f'/{attack_name}'):
				os.makedirs(self.path + f'/{attack_name}')
			f = open(self.path + f'/{attack_name}' + f"/{metrics[i]}_{'' if run_num == None else run_num}.txt", "a")
			f.write(f"{img_id}: {data[i]}"+ '\n')
			f.close()
			self.ave_metrics[metrics[i]] += data[i]
			if "all" in metrics[i]:
				continue
			summary.write(f"{metrics[i]}: {data[i]}"+ '\n')

		summary.write('\n')
		summary.close()
	
	def log_fitness(self, all_fitness_scores):
		if not os.path.exists(self.path + '/plots'):
			os.makedirs(self.path + '/plots')
		plt.plot([i for i in range(len(all_fitness_scores))], all_fitness_scores)
		plt.ylabel('Fitness')
		plt.xlabel('Generation')
		plt.savefig(self.path + '/plots/plt.png', bbox_inches='tight')
		plt.close()
	
	def log_individuals(self, individuals):
		if not os.path.exists(self.path + '/imgs'):
			os.makedirs(self.path + '/imgs')
		for i in range(len(individuals)):
			ad_img = transforms.ToPILImage()(individuals[i][0])
			draw = ImageDraw.Draw(ad_img)
			draw.text(xy=(10, 10), text=str(individuals[i][1]), fill=(255, 0, 0))
			ad_img.save(self.path + f'/imgs/img_{i}.png')

	def save_ave_metrics(self):
		for key, value in self.ave_metrics.items():
			self.ave_metrics[key] = value / self.size
			if not os.path.exists(self.path):
				os.makedirs(self.path)
			f = open(self.path + f"/{key}_ave.txt", "a")
			f.write(f"{self.ave_metrics[key]}"+ '\n')
			f.close()

	def plot_fitness(self, all_fitness_scores, generation=None):
		"""
		Plots the fitness scores over generations.

		Parameters:
		- all_fitness_scores: List of fitness scores for each generation.
		- generation: Optional, the current generation number to be annotated on the plot.
		"""
		
		if not os.path.exists(self.path + '/plots'):
			os.makedirs(self.path + '/plots')

		plt.figure(figsize=(10, 5))
		plt.plot(all_fitness_scores, label='Fitness Score')
		plt.title('Fitness over Generations')
		plt.xlabel('Generation')
		plt.ylabel('Fitness Score')
		plt.legend()
		if generation is not None:
			plt.annotate(f'Generation: {generation}', xy=(generation, all_fitness_scores[generation]),
							xytext=(generation, max(all_fitness_scores)*0.8),
							arrowprops=dict(facecolor='black', shrink=0.05))
		plt.grid(True)
		# plt.tight_layout()
		# plt_path = os.path.join(self.path, '/plots/fitness_over_generations.png')
		plt_path = self.path + f'/plots/fitness_over_generations_{generation}.png'
		print("\n-----------------",self.path," -----------------\n")
		print("\n-----------------",plt_path," -----------------\n")
		# plt.savefig(self.path + f'/plots/fitness_over_generations_{generation}.png')
		plt.savefig(plt_path)
		plt.close()  # Close the figure to prevent it from displaying in the notebook or script output
		print(f"Fitness plot saved to {plt_path}")
		# os.chdir(mycwd)	 # go back where you came from

