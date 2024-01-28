# Imports
from typing import List, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.distributions as td

import numpy as np
from math import prod

from tqdm import tqdm 
from os import system
import os

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdversarialExampleAttack:
	# Initialize class
	def __init__(self, 
				img, 
				model, 
				processor, 
				N=16, 
				G=400,
				threshold=0.9,
				p_mut=0.024, 
				alpha=1, 
				beta=150, 
				delta=0.02,			# starting deviation rate 
				m_range=1, 
				temp=1,
				p_unmut=0.2, 
				adaptive=True,
				multi_objective=False,
				fitness_weights=[0.1,0.8,0.8], 
				adaptive_params=False,
				log=False,
				attack_name='',
				dataset_name='',
				stamp=1
		) -> None:
		'''
			N: Population size 16
			G: Generations 50
		'''
		self.img = img
		self.target_sizes = torch.tensor([self.img.size[::-1]]).to(device=device)
		self.model = model.to(device=device)
		self.processor = processor
		
		# EvoAttack parameters
		self.N = N
		self.G = G
		self.threshold = threshold
		self.p_mut = p_mut
		self.p_mut_min = p_mut
		self.alpha = alpha # Range of mutation
		self.alpha_min = alpha
		self.beta = beta
		self.delta = delta
		self.adaptive = adaptive
		self.multi_objective = multi_objective

		self.attack_fitness_weight = fitness_weights[0]
		self.disparity_fitness_weight = fitness_weights[1]
		self.distance_fitness_weight = fitness_weights[2]

		self.adaptive_params = adaptive_params
		self.log = log
		
		# GenAttack parameters
		self.m_range = m_range  # GenAttack Alpha
		self.temp = temp 

		self.p_unmut= p_unmut
		
		self.size = 0

		self.attack_name=attack_name
		self.dataset_name=dataset_name
		self.model_name=self.model.__class__.__name__

		self.stamp=stamp+1

	def reset(self, p_unmut, fitness_weights):
		self.p_unmut = p_unmut
		self.attack_fitness_weight = fitness_weights[0]
		self.disparity_fitness_weight = fitness_weights[1]
		self.distance_fitness_weight = fitness_weights[2]


	def _get_mask(self, population, results):
		mask = torch.zeros_like(population).to(device=device)
		for n in range(population.shape[0]):
			for box, _ in zip(results[n]['boxes'], results[n]['scores']):
				box = [round(i) for i in box.tolist()]

				# Mask consists of the max score of areas in the image
				mask[n, 0:, box[1]:box[3], box[0]:box[2]] = torch.clamp(mask[n, 0:, box[1]:box[3], box[0]:box[2]], min=1)

		# Adding more noise to areas with higher score
		return mask * self.p_mut

	def _map_solution(self) -> Tuple[torch.Tensor, List[int]]:
		tensor = transforms.ToTensor()(self.img)
		self.size = prod(tensor.shape)
		
		population = tensor.unsqueeze(0).repeat(self.N,1,1,1) # Repeat original image N times
		inputs = self.processor(images=transforms.ToPILImage()(tensor), return_tensors="pt").to(device=device)
		outputs = self.model(inputs) if type(inputs) == torch.Tensor else self.model(**inputs)
		inputs = inputs.to(device='cpu')

		# convert outputs (bounding boxes and class logits) to COCO API
		results = self.processor.post_process_object_detection(outputs, target_sizes=self.target_sizes, threshold=self.threshold)[0]
		self.objects = len(results['boxes'])
		self.pixels = 0
		for box in results['boxes']:
			box = [round(i) for i in box.tolist()]
			self.pixels += (box[3]-box[1]) * (box[2]-box[0])

		self.max_norm2_dist = torch.dist(torch.zeros_like(tensor), tensor, p=2)

		# Initially, all results come from initial image
		mask= self._get_mask(population, [results for _ in range(self.N)])
		
		# Return original image as tensor, population, and mask
		return tensor, population.to(device=device), mask, [results for _ in range(self.N)]

	# Regular mutation
	def _mutation(self, mask, alpha):
		bernoulli = td.Bernoulli(mask).sample().to(device=device)		
		deviation = self.delta * alpha
		noise = torch.FloatTensor(mask.shape).uniform_(-deviation, deviation).to(device=device)
		mutation = bernoulli * noise
		noise = noise.to(device='cpu')
		bernoulli = bernoulli.to(device='cpu')
		return mutation, bernoulli
	
	# EvoAttack
	def _mutation_adaptive(self, mask, results, alpha, generation):
		mask = torch.zeros_like(mask)
		for n in range(mask.shape[0]):
			for box, _ in zip(results[n]['boxes'], results[n]['scores']):
				box = [round(i) for i in box.tolist()]
				size = (box[3]-box[1]) * (box[2]-box[0])
				if self.beta >= generation:
					pixels = max(generation / 5, 1)
				else: 
					pixels = size * self.p_mut * (generation / self.beta)
				mask[n, 0:, box[1]:box[3], box[0]:box[2]] = torch.clamp(mask[n, 0:, box[1]:box[3], box[0]:box[2]], min=pixels/size)
				
		return self._mutation(mask, alpha)

	def _mutate_population(self, population, mask, alpha=1, results =[], generation=0, show_first=False) -> torch.Tensor:
		mutation, bernoulli = self._mutation(mask, alpha) if not self.adaptive else self._mutation_adaptive(mask, results, alpha, generation)
		population += mutation
		if show_first:
			results = transforms.ToPILImage()(population[0])
			results.show()
			self.img.show()

		return population, bernoulli
	
	def _unmutate_population(self, population, original, alpha=1, generation=0) -> torch.Tensor:
		tensor = original.unsqueeze(0).repeat(self.N,1,1,1)
		ne = torch.ne(original.to(device=device) , population.to(device=device) ) * self.p_unmut
		bernoulli = td.Bernoulli(ne).sample().to(device=device) 
		reverse = 1 - bernoulli
		# bernoulli = td.Bernoulli(ne).sample().to(device=device)
		population = reverse.to(device=device) * population.to(device=device) + bernoulli.to(device=device) * tensor.to(device=device)
		return population

	def _crossover(self, parents, population, mask, fitness):
		_, nchannels, h, w = population.shape
		fitness_pairs = fitness[parents.long()].view(-1, 2)
		
		prob = fitness_pairs[:, 0] / fitness_pairs.sum(1)
		
		# Which parent pair are we chosing 
		parental_bernoulli = td.Bernoulli(prob)  
		
		inherit_mask = parental_bernoulli.sample((nchannels * h * w,))  # [N-1, nchannels * h * w]
		inherit_mask = inherit_mask.view(-1, nchannels, h, w) * torch.ceil(mask.to(device='cpu'))  # Mask areas of interest
		
		parent_features = population[parents.long()]
		children = (inherit_mask.float() * parent_features[::2]) + ((1-inherit_mask).float() * parent_features[1::2])
		
		return children

	def _tournament(self, tensor, mask, curr_population, prev_population, fitness_scores, min_idx, categorical):
		best_individual = curr_population[min_idx].clone().detach()

		# starting from the second individual in the population
		parents = categorical.sample()
		next_population = self._crossover(parents, curr_population.to(device='cpu'), mask, fitness_scores)
		
		return next_population.to(device=device), torch.unsqueeze(best_individual, 0)

	def _fitness_scores(self, population, disable_tqdm=False) -> List[torch.Tensor]:
		fitness_scores = torch.tensor([])
		attack_fitness_scores = torch.tensor([])
		disparity_fitness_scores = torch.tensor([])
		all_results = []
		for individual in tqdm(population, disable=disable_tqdm):
		
			ind_inputs = self.processor(images=transforms.ToPILImage()(individual.to(device='cpu')), return_tensors="pt").to(device=device)
			ind_outputs = self.model(ind_inputs) if type(ind_inputs) == torch.Tensor else self.model(**ind_inputs)
			
			results = self.processor.post_process_object_detection(ind_outputs, target_sizes=self.target_sizes, threshold=self.threshold)[0]
			all_results += [results]
			
			attack_fitness = sum(results["scores"]) 
			disparity_fitness = torch.dist(individual, transforms.ToTensor()(self.img).to(device=device), p=0)
			distance_fitness = torch.dist(individual, transforms.ToTensor()(self.img).to(device=device), p=2)
			if self.objects == 0:
				# mask = torch.zeros_like(population).to(device=device)
				fitness= 0 
			else:
				fitness = (attack_fitness / self.objects) if not self.multi_objective else (self.attack_fitness_weight * (attack_fitness / self.objects)) + (self.disparity_fitness_weight * (disparity_fitness / self.pixels)) + (self.distance_fitness_weight * (distance_fitness / self.max_norm2_dist))

			attack_fitness_scores = torch.cat((attack_fitness_scores, torch.tensor([attack_fitness])))
			disparity_fitness_scores = torch.cat((disparity_fitness_scores, torch.tensor([disparity_fitness])))
	
			fitness_scores = torch.cat((fitness_scores, torch.tensor([fitness])))
		
		mask = self._get_mask(population, all_results)
		if self.multi_objective:
			return fitness_scores, attack_fitness_scores, disparity_fitness_scores, mask, all_results
		return attack_fitness_scores, attack_fitness_scores, disparity_fitness_scores, mask, all_results
	
	# Run algorithm
	def find_adversarial_example(self, _curr=[], _len=[]):
	
		# Map to solution space 
		tensor, prev_population, mask, results = self._map_solution()

		# Initialize population
		if not _len == []:
			print(f'{self.attack_name} - {self.stamp}')				
			print(f'Dataset {_curr[0]} out of {_len[0]}')
			print(f'Model {_curr[2]} out of {_len[2]}')
			print(f'Sample {_curr[1]} out of {_len[1]}')
			print(f'Algorithm {_curr[4]} out of {_len[4]}')
			print(f'Run {_curr[3]} out of {_len[3]}')
			print()
			
		print(f'Generation 0 of {self.G}')
		curr_population, _ = self._mutate_population(prev_population, mask, results=results)

		fitness_scores, a_fitness_scores, _, mask, all_results = self._fitness_scores(curr_population)
		min_idx = np.argmin(fitness_scores)

		lo = torch.clamp(torch.norm(tensor, float('-inf')) - self.delta, min=0).to(device=device)
		hi = torch.clamp(torch.norm(tensor, float('inf')) + self.delta, max=1).to(device=device)
		
		all_fitness_scores = []
		all_best_individuals = []
		all_L0=[]
		all_L2=[]
		plateau_fitness=0
		plateau_i=0
		if self.objects ==0:
			return curr_population.to(device='cpu'), min_idx.to(device='cpu'), tensor.to(device='cpu'), 0, all_fitness_scores, all_best_individuals, all_L0, all_L2
		for i in range(self.G):	
			system('cls' if os.name == 'nt' else 'clear')
			if not _len == []:
				print(f'{self.attack_name} - {self.stamp}')
				print(f'Dataset {_curr[0]} out of {_len[0]}') if not _len[0] == -1  else None
				print(f'Model {_curr[2]} out of {_len[2]}') if not _len[2] == -1  else None
				print(f'Sample {_curr[1]} out of {_len[1]}') if not _len[1] == -1  else None
				print(f'Algorithm {_curr[4]} out of {_len[4]}') if not _len[4] == -1  else None
				print(f'Run {_curr[3]} out of {_len[3]}') if not _len[3] == -1  else None
				print()

			print(f'Generation {i+1} of {self.G}')
			print(f'Weights i: {round(self.attack_fitness_weight, 3)} - ii: {round(self.disparity_fitness_weight, 3)} - iii: {round(self.distance_fitness_weight, 3)}') if self.multi_objective else None
			indx = (fitness_scores == sorted(fitness_scores)[1]).nonzero(as_tuple=True)[0]
			print(f'Fit:   {fitness_scores[min_idx]} - 2nd: {fitness_scores[indx][0].item()}')
			print(f'A Fit: {a_fitness_scores[min_idx]} - 2nd: {a_fitness_scores[indx][0].item()}')
			l0 = torch.dist(tensor, curr_population[min_idx].to(device='cpu'), p=0).item()
			l2 = torch.dist(tensor, curr_population[min_idx].to(device='cpu'), p=2).item()
			all_L0 += [l0]
			all_L2 += [l2]
			print(f"L0:  {l0} - 2nd {torch.dist(tensor, curr_population[indx].to(device='cpu')[0], p=0).item()}")
			print(f"L2:  {round(l2, 2)} - 2nd {round(torch.dist(tensor, curr_population[indx].to(device='cpu')[0], p=2).item(),2)}")

			# Tournament selection & crossover
			categorical = td.Categorical(fitness_scores[None, :].expand(2 * self.N, -1))
			
			next_population, best_individual = self._tournament(tensor, mask, curr_population, prev_population, fitness_scores, min_idx, categorical)
			best_fitness_score = torch.unsqueeze(fitness_scores[min_idx].clone().detach(), 0)
			best_a_fitness_score = torch.unsqueeze(a_fitness_scores[min_idx].clone().detach(), 0)
			# Mutation
			next_population, _ = self._mutate_population(next_population, mask, alpha=self.m_range, results=all_results, generation=i)

			if self.multi_objective:
				if abs(best_fitness_score - plateau_fitness) > 0.001:
					plateau_fitness = best_fitness_score
					plateau_i = 0
				plateau_i += 1
				if (plateau_i) % 10 == 0:
					self.attack_fitness_weight = min(1., self.attack_fitness_weight + 0.05)
					self.disparity_fitness_weight = max(0.01 if not self.disparity_fitness_weight == 0. else 0., self.disparity_fitness_weight - 0.05)
					self.distance_fitness_weight = max(0.01 if not self.distance_fitness_weight == 0. else 0., self.distance_fitness_weight - 0.05)
					if not(self.attack_fitness_weight == 1. or self.disparity_fitness_weight == 1. or self.distance_fitness_weight == 1.):
						best_fitness_score = torch.tensor([self.attack_fitness_weight + self.disparity_fitness_weight + self.distance_fitness_weight])
					plateau_fitness = best_fitness_score
					self.p_unmut = max(.02, self.p_unmut - .02)
				next_population = self._unmutate_population(next_population, tensor, alpha=self.m_range, generation=i)


			# clip to ensure the distance constraints
			next_population  = torch.clamp(next_population, min=lo, max=hi)

			# Calculate population fitness scores
			fitness_scores, a_fitness_scores, _, mask, all_results = self._fitness_scores(next_population)
			max_idx = np.argmax(fitness_scores)

			next_population = torch.cat([best_individual, next_population[0:max_idx], next_population[max_idx+1:]], 0)
			fitness_scores = torch.cat([best_fitness_score, fitness_scores[0:max_idx], fitness_scores[max_idx+1:]], 0)
			a_fitness_scores = torch.cat([best_a_fitness_score, a_fitness_scores[0:max_idx], a_fitness_scores[max_idx+1:]], 0)
			min_idx = np.argmin(fitness_scores) 
			
			# Sorting and evaluation
			all_fitness_scores += [a_fitness_scores[min_idx]]
			all_best_individuals += [[next_population[min_idx], a_fitness_scores[min_idx]]] if i % (self.G // 5) == 0 else []
			
			if a_fitness_scores[np.argmin(a_fitness_scores)] == 0:
				fitness_scores, a_fitness_scores, _, mask, all_results = self._fitness_scores(next_population)
				min_idx = np.argmin(a_fitness_scores)

				if self.log:
					if not os.path.exists(f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/'):
						os.makedirs(f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/')
					
					im_stamp = len(os.listdir(f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/'))
					save_image(curr_population[min_idx], f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/imgs{im_stamp}.png')
				return next_population.to(device='cpu'), min_idx.to(device='cpu'), tensor.to(device='cpu'), i, all_fitness_scores, all_best_individuals, all_L0, all_L2

			prev_population = curr_population.clone().detach()  # update previous generation
			curr_population = next_population.clone().detach()  # update current generation
				
		_, a_fitness_scores, _, _, _ = self._fitness_scores(curr_population)
		min_idx = np.argmin(a_fitness_scores)

		if self.log:
			if not os.path.exists(f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/'):
				os.makedirs(f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/')
			
			im_stamp = len(os.listdir(f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/'))
			save_image(curr_population[min_idx], f'imgs/AEs/{self.stamp}/{self.model_name}/{self.dataset_name}/{self.attack_name}/imgs{im_stamp}.png')

		return curr_population.to(device='cpu'), min_idx.to(device='cpu'), tensor.to(device='cpu'), self.G, all_fitness_scores, all_best_individuals, all_L0, all_L2

