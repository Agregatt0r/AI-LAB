# Matrix multiplication

Multiply two matrices. Result entry `(i, j)` is the **dot product** of row `i` of A
and column `j` of B.

You can multiply A and B only if:
**columns of A = rows of B**

If A is `(m × n)` and B is `(n × p)`, then `A * B` has shape `(m × p)`.

```
A = [1  2]      B = [5  6]
    [3  4]          [7  8]

AB = [1*5+2*7   1*6+2*8] = [19  22]
     [3*5+4*7   3*6+4*8]   [43  50]
```
