import random
dis=[0,0,0,0,0]
for i in range(100000000):
    x=random.randint(1,6)
    y=random.randint(1,6)
    z=random.randint(1,6)
    dis[(x+y+z)%4]=dis[(x+y+z)%4]+1
for i in dis:
    print(i)