operations = ['+', '-', '*', '/']
prec = {"+": 0, "-": 0, "*": 1, "/": 1}

exp = ["(", "1", "+", "2", ")","/","6"]
exp = exp[::-1]    # ) 2 + 1 (
print(exp)

operations_stack = list()
operands_stack = list()


def ins(expression):
    while len(exp) != 0:
        if expression[0] == ")":
            operations_stack.append(expression[0])
            expression.pop(0)
            print("expression1: ", expression)
            return ins(expression)

        elif expression[0] == "(":
            expression.pop(0)
            operands_stack.insert(0,operations_stack.pop(-1))
            operations_stack.pop(0)
            return ins(expression)

        elif expression[0] not in prec:
            print("true")
            operands_stack.append(expression[0])
            expression.pop(0)
            print("expression2: ", expression)
            return ins(expression)

        elif expression[0] in prec:
            # if operations[expression[0]] < operations[]
            operations_stack.append(expression[0])
            print("operations: ", operations_stack)
            expression.pop(0)
            print("expression3: ", expression)
            return ins(expression)


    else:
        print("END")
        return expression

x = ins(exp)
print("operations stack: ", operations_stack)
print()
print("operand stack: ", operands_stack)
print()
print("Expression: ", exp)
