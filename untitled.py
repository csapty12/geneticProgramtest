def main2():
	split_parents = [['(', 'X1', '+', 'X1', '+', '5', ')', '-', '18', '+', '17', '+', '10', 'end'], ['(', '(', 'X1', '*', '13', ')', '*', '18', '*', '6', ')', 'end']]
	get_prefix_parents = get_prefix_notation(split_parents)
	print(get_prefix_parents)
	parent_tree1 = get_prefix_parents[0]
	parent_tree2 = get_prefix_parents[1]

    # make the parent trees 
	make_parent_tree_one = make_tree(parent_tree1)
	make_parent_tree_two = make_tree(parent_tree2)

	show_parent_tree_one = print_full_tree(make_parent_tree_one[0])
	show_parent_tree_two = print_full_tree(make_parent_tree_two[0])

    # print(show_parent_tree1)
    # print(show_parent_tree2)
	make_child_tree_one = copy.deepcopy(make_parent_tree_one)
	show_child_tree_one = print_full_tree(make_child_tree_one[0])

	make_child_tree_two = copy.deepcopy(make_parent_tree_two)
	show_child_tree_two = print_full_tree(make_child_tree_two[0])

	select_child_node_one = select_random_val(make_child_tree_one[1])
	select_child_node_two = select_random_val(make_child_tree_two[1])

	print("selected node 1: ", select_child_node_one)
	print("selected node 2: ", select_child_node_two)

	random_node_one = find_subtree(make_child_tree_one[0],make_child_tree_one[1],select_child_node_one)
	random_node_two = find_subtree(make_child_tree_two[0],make_child_tree_two[1],select_child_node_two)
	
	print('swapping: ', random_node_one.value, random_node_one.nodenum , " with ",random_node_two.value, random_node_two.nodenum)
	FullTree.swap_nodes(random_node_one, random_node_two)

	print("parent 1: ")
	print(show_parent_tree_one)
	print('\n')
	print("parent 2: ")
	print(show_parent_tree_two)
	print('\n')
	print('makeing new tree 1: ')
	print(show_child_tree_one)
	print('\n')
	print('make new tree 2: ')
	print(show_child_tree_two)
	print('\n')