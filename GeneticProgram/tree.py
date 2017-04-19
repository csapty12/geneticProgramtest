from node import Node
from ExpressionGenerator import GenMember
from random import choice


class Tree(object):
    """
    Class to make a tree, find a subtree, and and perform genetic operations on the tree.
    """
    def __init__(self, root_node=None):
        """
        Constructor to initialise the root node of a tree.
        :param root_node: None until value given to the node.
        """
        self.root = root_node

    def make_tree(self, pref_list):
        """
        Function to build the tree structure using the prefix expression
        :param pref_list: prefix list
        :return: root node, list of nodes and node ID's
        """

        nodes = list()
        nodenums = list()
        root_node = Node(pref_list[0])

        nodes.append(root_node)
        nodenums.append(root_node.nodenum)

        current_node = root_node  # use current node to point the current being being used.
        pref_list.pop(0)

        while len(pref_list) > 0:
            if current_node.value in GenMember.operations:
                if current_node.left_child is None:
                    current_node.add_child(pref_list[0], left=True)  # add a left child with its value
                    pref_list.pop(0)
                    current_node = current_node.left_child
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

                elif current_node.left_child is not None and current_node.right_child is not None:
                    current_node = current_node.parent
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

                else:
                    current_node.add_child(pref_list[0], left=False)
                    pref_list.pop(0)
                    current_node = current_node.right_child
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

            elif current_node.value not in GenMember.operations:
                current_node = current_node.parent

                if current_node.left_child is not None and current_node.right_child is not None:
                    current_node = current_node.parent
                    nodes.append(current_node)
                    nodenums.append(current_node.nodenum)

        return root_node, nodes, nodenums

    def print_full_tree(self, tree):
        """
        Function to print out the the tree in the tree structure, list of nodes or the node ID's
        :param tree: the tree
        :return: the representation of the tree.
        """
        return tree

    def find_subtree(self, tree, list_nodes, rnd_val):
        """
        Function to find a subtree within the tree to ensure that subtree exists. Function uses a depth first
        search to locate the subtree.
        :param tree: the tree to search
        :param list_nodes: list of nodes of that tree.
        :param rnd_val: the random value selected which find subtree is searching for.
        :return: the node that has been located within the subtree.
        """
        current_node = tree
        if current_node.value == rnd_val[0] and current_node.nodenum == rnd_val[1]:
            current_node.checked = True
            return current_node

        else:
            # if the current node left child exists:
            if current_node.left_child is not None and current_node.left_child.checked is False:
                # mark the current node as checked
                current_node.checked = True
                # move into the left child node.
                current_node = current_node.left_child
                return self.find_subtree(current_node, list_nodes, rnd_val)

            else:
                # if the curent node left child doesnt exist i.e is a leaf node
                current_node.checked = True
                # move to the parent
                if current_node.right_child is not None and current_node.right_child.checked is False:
                    current_node.checked = True
                    current_node = current_node.right_child
                    return self.find_subtree(current_node, list_nodes, rnd_val)

                else:
                    current_node = current_node.parent
                    # if the current node left and right child both have been cheked, move to the curren node parent
                    if current_node.left_child.checked is True and current_node.right_child.checked is True:
                        current_node = current_node.parent
                        return self.find_subtree(current_node, list_nodes, rnd_val)

                    else:
                        # move pointer to the right child
                        current_node = current_node.right_child
                        return self.find_subtree(current_node, list_nodes, rnd_val)

    def select_random_val(self, list_nodes):
        """
        Function to select a random node value from the list of nodes.
        :param list_nodes: list of nodes
        :return: the selected node value, the selected node ID, the selected node.
        """

        # pop the root node out to prevent root node being selected.
        root = list_nodes[0].nodenum
        x = list_nodes.pop(0)
        while True:
            y = choice(list_nodes)
            if y.nodenum != root:
                break
        list_nodes.insert(0, x)
        return y.value, y.nodenum, y

    def swap_nodes(self, tree_one, tree_two, node_one, node_two):
        """
        Function to take two trees and their selected subtrees to swap them over to simulate genetic crossover.
        :param tree_one: parent tree one
        :param tree_two: parent tree two
        :param node_one: parent tree one randomly selected node
        :param node_two: parent tree two randomly selected node
        :return: child tree one, child tree two.
        """

        # get the parents of each node selected
        node_one_parent = node_one.parent
        node_two_parent = node_two.parent

        # check value and nodenum to ensure correct subtree is being swapped.
        if node_one_parent.left_child.value == node_one.value \
                and node_one_parent.left_child.nodenum == node_one.nodenum:
            node_one_parent.left_child = node_two

            node_one_parent.left_child.parent = node_one_parent

        else:
            node_one_parent.right_child = node_two

            node_one_parent.right_child.parent = node_one_parent

        if node_two_parent.left_child.value == node_two.value and node_two_parent.left_child.nodenum == \
                node_two.nodenum:
            node_two_parent.left_child = node_one
            node_two_parent.left_child.parent = node_two_parent

        else:
            node_two_parent.right_child = node_one
            node_two_parent.right_child.parent = node_two_parent

        return tree_one, tree_two

    def mutate_node(self, tree, list_nodes, node):
        """
        Function to mutate the randomly selected node based on its current value and arity
        :param tree: tree to be mutated
        :param list_nodes: list of nodes associated with the tree
        :param node: the node selected for mutation
        :return: the updated tree, the updated list of nodes associated with the tree
        """

        # check if node selected is an operator.
        if node.value in ['+', '-', '*', '/']:
            # select operator based on same arity of node to be changed
            node.value = choice(['+', '-', '*', '/'])
            return tree, list_nodes  # return the new tree, new list_nodes, new mutated node.

        else:
            # check if terminal value and not a variable
            if node.value not in ["X1", "X2", "X3", "X4", "X5"]:
                # alter the value by a small amount
                val = float(node.value)
                val -= 0.1
                node.value = str(val)

            else:
                # if value is a variable, then select another variable
                node.value = choice(["X1", "X2", "X3", "X4", "X5"])

            return tree, list_nodes

    def get_child_one(self, child_one):
        """
        Function to get the first child that is produced
        :param child_one: the child produced
        :return: the first child
        """
        return child_one

    def get_child_two(self, child_one, child_two):
        """
        Function to get the second child independently
        :param child_one: the first child
        :param child_two: the second child
        :return: only the values of the second child.
        """
        return child_two[len(child_one):]

    def make_list_nodes(self, tree, l1=list()):
        """
        Function to make a list of nodes based on the tree that has been inputted
        :param tree: the tree to be converted to a list of nodes.
        :param l1: empty list which is appended to recursively.
        :return: the list of nodes
        """
        root_node = tree
        current_node = root_node

        if current_node.checkedAgain is True and current_node.parent is None and current_node.left_child.checkedAgain \
                is True and current_node.right_child.checkedAgain is True:
            return l1

        else:
            if current_node.left_child is not None and current_node.left_child.checkedAgain is False:
                current_node.checkedAgain = True
                l1.append(current_node)
                current_node = current_node.left_child
                return self.make_list_nodes(current_node)

            else:
                current_node.checkedAgain = True
                if current_node.right_child is not None and current_node.right_child.checkedAgain is False:
                    current_node = current_node.right_child
                    return self.make_list_nodes(current_node)

                else:
                    if current_node not in l1:
                        l1.append(current_node)

                    current_node = current_node.parent
                    if current_node.left_child.checkedAgain is True and current_node.right_child.checkedAgain is True \
                            and current_node.parent is not None:
                        current_node = current_node.parent
                        return self.make_list_nodes(current_node)

                    elif current_node.left_child.checkedAgain is True and current_node.right_child.checkedAgain \
                            is True and current_node.parent is None:
                        return self.make_list_nodes(current_node)

                    else:
                        current_node = current_node.right_child
                        return self.make_list_nodes(current_node)

    def build_child(self, tree, list_nodes):
        """
        function to return the children trees and list of nodes.
        :param tree: child tree
        :param list_nodes: list of nodes for that child tree.
        :return: tuple (tree, list of nodes)
        """
        return tree, list_nodes

