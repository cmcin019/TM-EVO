import fiftyone.zoo as foz

def get_coco_data(dataset_dir):
	dataset = foz.load_zoo_dataset(
		'coco-2017', 
		split="test", 
		label_types=["detections"], 
		dataset_dir=dataset_dir+"/Datasets/COCO/fiftyone",
		dataset_name="coco_test-detection-set"
		)

	dataset.persistent = True
	return 'coco-2017', dataset

def get_kitti_data(dataset_dir):
	dataset = foz.load_zoo_dataset(
		'kitti', 
		split="test", 
		label_types=["detections"], 
		dataset_dir=dataset_dir + "/Datasets/KITTI/fiftyone",
		dataset_name="kitti-test-detection-set"
		)

	dataset.persistent = True
	return 'kitti', dataset

def get_imagenet_data(dataset_dir):
	dataset = foz.load_zoo_dataset(
		'imagenet-2012', 
		split="validation", 
		label_types=["detections"], 
		source_dir=dataset_dir+"/Datasets/ImageNet",
		dataset_dir=dataset_dir+"/Datasets/ImageNet/fiftyone",
		dataset_name="imagenet-test-detection-set"
		)

	dataset.persistent = True
	return 'imagenet-2012', dataset
