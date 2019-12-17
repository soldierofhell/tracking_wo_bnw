import torch
from torch.utils.data import Dataset

from .football_sequence import Football_Sequence


class Football_Wrapper(Dataset):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dets, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['']
		test_sequences = ['DEMO-bayern_full',
				 'DEMO-poland-portugal',
				 'DEMO-poland-portugal_all',
				 'DEMO-poland-portugal_missingSet',
				 'DEMO-poland-portugal_orig',
				 'DEMO-poland-portugal_v3All',
				 'DEMO-polska-belgia',
				 'DEMOv2-atomowe',
				 'DEMOv2-plaku',
				 'FIFA-arsenal',
				 'FIFA-chelsea',
				 'FIFA-chelsea-2']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		else:
			raise NotImplementedError("MOT split not available.")

		self._data = []
		for s in sequences:
			self._data.append(Football_Sequence(seq_name=s, dets=dets, **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]
