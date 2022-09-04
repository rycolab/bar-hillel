from rayuela.base.termdep import TreeBank, Tree
import os

class Trees:
	def __init__(self, path):
		self.path = path

	def importFile(self):
		abs_path = os.path.abspath(self.path)
		tree_bank = TreeBank(abs_path)
		trees = [(Tree(t), dep_label, pos, pos_plus, id) for t, dep_label, pos, pos_plus, id in tree_bank.generator()]
		return trees

	def format(self, tr):
		tr = list(tr)
		# TODO: make cleaner
		tr = tuple([-1 if i == -1 else i+1 for i in list(tr)])
		return tr

	def get_tuples(self):
		trees = self.importFile()
		all_trees = []
		all_feats = []
		for tree, dep_label, pos, pos_plus, id in trees:
			tree_dic = {}
			feat_dic = {}
			for tr, d_l, p, p_p in zip(tree.tree, dep_label, pos, pos_plus):
				tr = self.format(tr)
				tree_dic[tr] = None
				feat_dic[tr[1]] = [d_l, p, p_p]
			tree_dic["max_len"] = len(tree.tree)
			tree_dic["id"] = id
			all_trees.append(tree_dic)
			all_feats.append(feat_dic)
		return all_trees, all_feats