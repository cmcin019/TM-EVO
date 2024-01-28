# Imports
import torch

from PIL import Image

from os import system
import os
import time
import argparse

import models
from logger import logger
from AdversarialExampleAttack import AdversarialExampleAttack
from data import get_kitti_data, get_coco_data

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

is_adapt = True

def experiment(model=['faster'], datasets=['coco', 'kitti'], dataset_size=5, algs=['evo'], num_per_run=3, N=32, G=400, p_mut=0.024, alpha=1, beta=150, delta=0.0002, p_unmut=0.2, fitness_weights=[0.1,0.8,0.8], seed=6):
	system('cls' if os.name == 'nt' else 'clear')

	model_list=[]
	if 'deter' in model:
		# DETR-RESNET50
		model_list += [models.get_deter()]

	if 'faster' in model:
		# FASTER-CNN-RESNET50
		model_list += [models.get_FasterRCNN()]

	dataset_list = []
	if 'coco' in datasets:
		dataset_list += [get_coco_data(dataset_dir=dataset_dir)]

	if 'kitti' in datasets:
		dataset_list += [get_kitti_data(dataset_dir=dataset_dir)]


	with torch.no_grad():
		d_i = 0
		ave_metrics = {'norm_0':0, 'norm_2':0, 'generations': 0, 'run_time':0, 'fitness':0, 'all_fitness':[], 'all_norm_0':[], 'all_norm_2':[]}
		stamp=len(os.listdir('logs/')) if os.path.exists(f'logs/') else 0
		log = logger('', '', is_adapt, ave_metrics, dataset_size, max_G=G, stamp=stamp)
		log.log_parameters([["N",N], ["G",G], ["p_mut",p_mut], ["alpha",alpha], ["beta",beta], ["delta",delta], ["p_unmut",p_unmut], ['attack fitness weight',fitness_weights[0]],['disparity fitness weight',fitness_weights[1]],['distance fitness weight',fitness_weights[2]]])
		for dataset_name, dataset in dataset_list:
			d_i += 1
			# Choose a random subset of samples to add predictions to
			predictions_view = dataset.take(dataset_size, seed=seed)
			m_i = 0
			for processor, model, id2label in model_list:
				m_i+=1
				# for j in range(num_per_run):
				log.set_new_info(dataset=dataset_name, model=model.__class__.__name__, ave_metrics=ave_metrics)
				i=0
				for sample in predictions_view:
					i += 1
					img = Image.open(sample.filepath)

					attack_list = []
					if 'evo' in algs:
						attack_list += [AdversarialExampleAttack(img, model, processor, N=N, G=G, p_mut=p_mut, multi_objective=False, beta=beta, delta=delta, p_unmut=p_unmut, adaptive=is_adapt, fitness_weights=fitness_weights, log=True, attack_name='evo',    dataset_name=dataset_name, stamp=stamp)]
					if 'evo_doo' in algs:
						attack_list += [AdversarialExampleAttack(img, model, processor, N=N, G=G, p_mut=p_mut, multi_objective=True,  beta=beta, delta=delta, p_unmut=p_unmut, adaptive=is_adapt, fitness_weights=fitness_weights, log=True, attack_name='evo_doo', dataset_name=dataset_name, stamp=stamp)]					
					if 'evo_too' in algs:
						attack_list += [AdversarialExampleAttack(img, model, processor, N=N, G=G, p_mut=p_mut, multi_objective=True,  beta=beta, delta=delta, p_unmut=p_unmut, adaptive=is_adapt, fitness_weights=fitness_weights, log=True, attack_name='evo_too', dataset_name=dataset_name, stamp=stamp)]					
					
	
					a_i =0
					for attack in attack_list:
						a_i+=1
						for j in range(num_per_run):
							attack.reset(p_unmut=p_unmut, fitness_weights=fitness_weights)

							start_attack_time = time.time()
							ad_img, best, original, generations, all_fitness_scores, _, all_L0, all_L2 = attack.find_adversarial_example([d_i,i, m_i, j+1, a_i], [len(datasets), dataset_size, len(model_list), num_per_run, len(attack_list)])
							run_time = round(time.time() - start_attack_time,3)

							# TODO: Metrics
							norm_0 = torch.dist(original, ad_img[best], p=0).item()
							norm_2 = torch.dist(original, ad_img[best], p=2).item()
							
							lst_mets = ['norm_0', 'norm_2', 'generations', 'run_time', 'fitness', 'all_fitness', 'all_norm_0', 'all_norm_2']
							lst_data = [norm_0, norm_2, generations, run_time, 0 if all_fitness_scores == [] else all_fitness_scores[-1], all_fitness_scores, all_L0, all_L2]
							log.log_metric(lst_mets, os.path.basename(sample.filepath), lst_data, run_num=j+1, attack_name=attack.attack_name)
							print("l0 norms (changed pixels): ", int(norm_0))
							print("l2 norms (Degree of perturbations): ", round(norm_2, 3))

							torch.cuda.empty_cache()


def main(args):
	f_w = [.1,.9,.0]
	f_w_tm = [.1,.9,.9]

	if args.algs == "EVO":
		algs=["evo"]
	elif args.algs == "TM_EVO":
		algs=["evo_too"]
	else:
		algs=["evo","evo_too"]
	
	if args.datasets == "COCO":
		datasets=["coco"]
	elif args.datasets == "KITTI":
		datasets=["kitti"]
	else:
		datasets=["coco","kitti"]

	if args.mdels == "DETR":
		mdels=["deter"]
	elif args.mdels == "FASTER":
		mdels=["faster"]
	else:
		mdels=["deter","faster"]

	for alg in algs:
		for dataset in datasets:
			for model in mdels:
				torch.cuda.empty_cache()
				experiment(model=[model], datasets=[dataset], dataset_size=args.dataset_size, algs=[alg], num_per_run=args.runs, N=args.population, G=args.generations, p_mut=args.p_mut, alpha=args.alpha, beta=args.beta, delta=args.delta, p_unmut=args.p_unmut, fitness_weights=f_w if alg=="evo_too" else f_w_tm, seed=args.seed)
	torch.cuda.empty_cache()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data_location", help="Enter directory to save datasets", type=str)

	parser.add_argument("-a", "--algs", help="Enter video number (EVO, TM_EVO, ALL)", type=str, default="all")
	parser.add_argument("-d", "--datasets", help="Enter video number (COCO, KITTI or ALL)", type=str, default="all")
	parser.add_argument("-m", "--mdels", help="Enter models to experiment (DETR, FASTER or ALL)", type=str, default="all")

	parser.add_argument("-ds", "--dataset_size", help="Number of images to process", type=int, default=10)
	parser.add_argument("-r", "--runs", help="Runs per experiment", type=int, default=5)
	parser.add_argument("-p", "--population", help="Population size", type=int, default=32)
	parser.add_argument("-g", "--generations", help="Max generations (at least 5)", type=int, default=600)
	parser.add_argument("-pm", "--p_mut", help="Mutation probability", type=float, default=0.012)
	parser.add_argument("-pu", "--p_unmut", help="Mutation reduction probability", type=float, default=0.3)
	parser.add_argument("-al", "--alpha", help="Alpha hyperparameter", type=int, default=1)
	parser.add_argument("-be", "--beta", help="Beta hyperparameter", type=int, default=75)
	parser.add_argument("-de", "--delta", help="Delta hyperparameter", type=float, default=0.4)
	parser.add_argument("-s", "--seed", help="Seed for dataset", type=int, default=6)

	args = parser.parse_args()
	dataset_dir=args.data_location

	start_time = time.time()
	main(args)
	print("--- %s seconds ---" % round(time.time() - start_time))
	
	
