def transpose(A):
    At = []
    nCol = len(A[1])
    for i in range(nCol):
        r = []
        for row in A:
            r.append(row[i])
        At.append(r)
    return At


def array_mult(A, B):
    nRow = len(A)
    nCol = len(B[1])
    
    C = []
    for i in range(nRow):
        row = []
        for j in range(nCol):
            row.append(sum([x*y for x,y in zip(A[i], (transpose(B))[j])])) 
        C.append(row)
    return C

M1 = [[1, 2, 3], [-2, 3, 7]]
M2 = [[1,0,0],[0,1,0],[0,0,1]]
print(transpose(M2))
print(array_mult(M1,M2))

# [[1, 2, 3], [-2, 3, 7]]