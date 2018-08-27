## Knn Algorithm Pseudocode:

- Calculate $d(x, x_i) i = 1, 2, \dots , n$ where d denotes the Euclidean distance between the points.
- Arrange the calculated n Euclidean distances in non-decreasing order.
- Let $k > 0$, take the first k distances from this sorted list.
- Find those k-points corresponding to these k-distances.
- Let $k_i$ denotes the number of points belonging to the ith class among k points i.e. $k \geq 0$
- If $k_i > k_j \forall i \neq j$ then put x in class i.

