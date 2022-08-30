a = [[3,6,8,65,3],[3,6,8,65,3],[3,6,8,65,3]]
b = [34,2,5,3,5]
c= []
for x in range(len(a)):
    c.append([(x*1.0)/y for x, y in zip(a[x], b)])

print(c)

test = '1.2'
#ftest = float(test)
print(int(float(test)))