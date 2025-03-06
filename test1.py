def convolution(n,k):
    if n==k:
        return 1
    if k==1:
        return n
    else :
        return convolution(n-1,k)+convolution(n-1,k-1)
print(convolution(100,50))