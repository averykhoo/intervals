import sys; sys.path[:0]=['.', 'archive/v1']
import intervals as v2
A = v2.MultiInterval.parse('{ [0, 1] , (2, inf] }')
print(A.pieces, type(A.pieces[0]))
p = A.pieces[1]; print(p.inf, p.sup, p.inf_closed, p.sup_closed)
print(A.cuts)
