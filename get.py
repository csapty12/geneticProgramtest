import numpy as np
def get_fitness(expression, child = False):
    print("expression: ", expression)
    row = [[ 0.18584133, 0.22987825, 0.15035332, 2.26796244 , 1.72085425],
     [ 0.16285377, 0.2936199 , 0.14842959, 2.1121061, 1.72671183]]
    
    labels = [0,0]

    row = np.asarray(row)
    print("row:", row, type(row))
    new_row = row.T
    X1 = new_row[0]
    X2 = new_row[1]
    X3 = new_row[2]
    X4 = new_row[3]
    X5 = new_row[4]
    print()
    print( new_row)
    print()
    test = list()
    for i in expression:
        x = eval(i)
        x = x.tolist()
        
        test.append(x)
    print(test)



   

    

get_fitness(['X1+4+X2+ X3/X4 *(6-X5)', 'X1+5'])


"""
this is the current form
[array([ 4.41571958,  4.45647367]), array([ 5.18584133,  5.16285377])]

I want it in the form:
[array([ 4.41571958, 5.18584133 ]), array([ 4.45647367,  5.16285377])]

"""