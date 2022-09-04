#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from _typeshed import Self
# from numpy.char import array
from numpy.lib.function_base import append
import pyconll
import numpy as np


class Tree(object):

    sym_tbl = {
        "empty":                    ' ',
        "node":                     'O',
        "h_edge":                   '━',
        "v_edge":                   '┃',
        "l_corner":                 '┏',
        "t_intersection":           '┳',
        "cross_intersection":       '╋',
        "r_corner":                 '┓',
        "projection":               '┆',
        "projection_intersection":  '┿'
    }

    def __init__(self, tree, root=-1, text=None):
        """ `text` is used for pretty printing only. """
        tree = list(tree)
        tree.sort(key=(lambda x: x[1]))
        self.tree = tuple(tree)
        self.root = root

        # text preprocessesing and checks
        # If no text is given, set "A B C ... a b c ... 0 1 2 ...", one word per node.
        if text is None:
            words = [chr(i) for i in range(65, 91)]  # upper case letters
            words += [chr(i) for i in range(97, 123)]  # lower case letters
            words += [str(i) for i in range(0, self.size-2*26)]  # numbers
            text = " ".join(words[:self.size])
        self.text = text.strip('. \t')

        # Make sure the text and node amount is equal
        words = self.text.count(" ") + 1
        if words != self.size:
            raise ValueError("Graph and text don't fit together. " +
                             f"Expected {words} nodes but got {self.size}")

        self.node_column = []
        acc = -1  # start at -1 because index starts at 0, and ceil gives at least 1
        for word in self.text.split(' '):
            self.node_column.append(int(acc + np.ceil(len(word)/2)))
            acc += len(word) + 1

    def _generate_matrix(self) -> "np.array":
        """
        Generate a suitable 2D matrix for pretty printing.
        `text` is already inserted as last row.
        """
        # Get tree part of the matrix in np.array form
        m, l, r, cd = self._tree_matrix()

        # make final matrix
        depth = len(m)
        rows = depth + 2
        columns = len(self.text)
        matrix = np.full((rows, columns), self.sym_tbl["empty"])

        # insert top part of matrix with tree using block matrix operation
        matrix[:depth, l:r] = m

        # draw projection lines
        matrix = self._add_projection_lines(matrix, cd, depth)

        # add text at the bottom
        matrix[-1] = np.array(list(self.text))

        return matrix

    @property
    def size(self) -> int:
        """
        Gives number of edges in tree
        Works, because every vertex only has one inbound edge.
        """
        return len(self.tree)

    @property
    def depth(self) -> int:
        """
        Return the depth of the tree (excluding artificial root node).
        Warning: Due to list representation of edges, this is slow.
        """
        res = self._dfs(self.root)
        return res - 1

    def _dfs(self, root) -> int:
        """
        Internal method that gives depth starting from root node.
        Warning: Due to list representation of edges, this is slow.
        """
        max_depth = 0
        for parent, node in self.tree:
            if parent == root:
                partial_depth = self._dfs(node)
                max_depth = max(max_depth, partial_depth)
        return max_depth + 1

    def _tree_matrix(self, root=None):
        """
        This function uses recursion to build each subtree from a given node.
        If no node is given, it is assumed that the root node is requested.
        This makes a matrix slightly bigger than it's children, checks if it is
        projective, and then returns a composite of the subtrees with the added symbols

        returns a matrix just big enough to house the subtree, the subtree left and right
        edges proportional to the final matrix, de depth of the subtree, and the depth
        each child is drawn at.
        """

        # If no root is specified, get first node with parent self.root
        if root is None:
            root = next(node for node in self.tree if node[0] == self.root)

        # get root position
        root_pos = self.node_column[root[1]]

        # Get children of node
        children = [node for node in self.tree if node[0] == root[1]]

        # If there are no children, then it is a leaf node
        if len(children) == 0:
            # Generate node matrix with correct depth
            matrix = np.array([self.sym_tbl["node"]])
            return matrix, root_pos, root_pos + 1, {root[1]: 0}

        # get matrices of children and get columns of children
        children_arr = [self._tree_matrix(node) for node in children]

        # check that children fit side to side
        prev_right = 0
        max_depth = 0
        for m, l, r, _ in children_arr:
            if l < prev_right:  # if there is overlap, add bellow previous
                max_depth += len(m)
            else:
                max_depth = max(max_depth, len(m))
            prev_right = r

        # find leftmost and rightmost node
        left_most = min(root_pos, min([l for _, l, _, _ in children_arr]))
        right_most = max(root_pos + 1, max([r for _, _, r, _ in children_arr]))

        # connect nodes:
        children_columns = [self.node_column[node] -
                            left_most for _, node in children]

        # create return matrix
        matrix = np.full((max_depth + 1, right_most -
                         left_most), self.sym_tbl["empty"])

        # iterate of children to add sub-matrices to resulting matrix and
        # add children depth to new depth dictionary
        prev_right = 0
        prev_depth_floor = 1
        prev_depth_ceil = 1
        children_depth = {}
        # definitions on how to draw vertical edges
        v_edges = {
            self.sym_tbl["empty"]:      self.sym_tbl["v_edge"],
            self.sym_tbl["h_edge"]:     self.sym_tbl["cross_intersection"]
        }

        for (m, l, r, cd), col in zip(children_arr, children_columns):
            # if subtree overlaps with previous tree, draw it lower, else
            # draw on the same level
            if l < prev_right:
                rows_floor, rows_ceil = prev_depth_ceil, prev_depth_ceil + \
                    len(m)

                # update depth floor and ceiling to new value
                prev_depth_floor = prev_depth_ceil
                prev_depth_ceil += len(m)
            else:  # else, draw it normally
                rows_floor, rows_ceil = prev_depth_floor, prev_depth_floor + \
                    len(m)

                # update ceiling to new value
                prev_depth_ceil = max(
                    prev_depth_floor + len(m), prev_depth_ceil)

            # add sub-matrices to resulting matrix
            matrix[rows_floor:rows_ceil, l - left_most:r - left_most] = m

            # update prev_right
            prev_right = r

            # add vertical edges
            if prev_depth_floor > 1:
                matrix[1:prev_depth_floor, col] = np.vectorize(
                    lambda x: v_edges[x])(matrix[1:prev_depth_floor, col])

            # add adjusted children depth to new depth
            children_depth.update(
                {k: v + prev_depth_floor for k, v in cd.items()})

        # find left and right ends of the edge
        edge_left = min(root_pos - left_most, min(children_columns))
        edge_right = max(root_pos - left_most, max(children_columns))

        # add horizontal edges
        matrix[0][edge_left:edge_right] = self.sym_tbl["h_edge"]

        # add intersections for nodes, and coners for the edges
        matrix[0][children_columns] = self.sym_tbl["t_intersection"]
        matrix[0][edge_left] = self.sym_tbl["l_corner"]
        matrix[0][edge_right] = self.sym_tbl["r_corner"]

        # put in node
        matrix[0][root_pos-left_most] = self.sym_tbl["node"]

        # add node to children_depth
        children_depth[root[1]] = 0

        return matrix, left_most, right_most, children_depth

    def _add_projection_lines(self, matrix, nodes, depth):
        """
        Add projection lines for each node
        """
        # helper function to convert projection lines
        projection_lines = {
            self.sym_tbl["empty"]:      self.sym_tbl["projection"],
            self.sym_tbl["h_edge"]:     self.sym_tbl["projection_intersection"]
        }

        def add_proj_line(elem):
            return projection_lines[elem]

        # add projection lines downwards from each node, including buffer
        for (node, row) in nodes.items():
            fr, lr = row+1, depth+1
            column = self.node_column[node]
            matrix[fr:lr, column] = np.vectorize(
                add_proj_line)(matrix[fr:lr, column])

        return matrix

    def is_projective(self) -> bool:
        """
        TODO: Remove this, the serious function is in algorithms.py.
        """
        string = self.__str__()
        return not (self.sym_tbl['projection_intersection'] in string or self.sym_tbl['cross_intersection'] in string)

    def __str__(self):
        """
        Generates a string from a matrix with every row as a line
        """
        matrix = self._generate_matrix()
        return '\n'.join([''.join(row) for row in matrix])

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.tree)

    def __getitem__(self, i):
        return self.tree[i]

    def __iter__(self):
        return iter(self.tree)


class TreeBank(object):

    def __init__(self, fin):
        self.trees = pyconll.load_from_file(fin)

    def generator(self):
        """
        Returns trees as a tuple of pairs, e.g.,
        ((2, 0), (2, 1), (-1, 2), (5, 3), (5, 4), (2, 5), (2, 6)),
        The ordering of the pairs does not matter.
        -1 is a distinguished integer for the root
        """

        sent_id = 0
        for sentence in self.trees:
            root = None
            broken = False
            dep = []
            dep_label = []
            pos = []
            pos_plus = []

            sent_id += 1

            for i, word in enumerate(sentence):

                if word.head is None:
                    broken = True
                    break

                head = int(word.head)-1
                dep.append((head, i))
                dep_label.append(word.deprel)
                pos.append(word.upos)
                pos_plus.append(word.upos+"_"+word.xpos)


                if head == -1:
                    root = dep

            try:
                dep = Tree(tuple(dep), root)
            except ValueError:
                continue

            if broken or root is None:
                continue

            yield dep, dep_label, pos, pos_plus, sent_id