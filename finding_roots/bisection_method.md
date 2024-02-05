# The main Idea
The main Idea for the bisection method comes from the
intermediate value theorem: if $f(a)$ and $f(b)$ have
different signs and $f$ is continous, then $f$ must
have a zero between $a$ and $b$. And by evaluating it 
at the midpoint, $c=\frac{1}{2}(a+b)$ we get that 
$f(c)$ is either zero, has the same sign as $f(a)$ or
the same sign as $f(b)$. If $f(c)$ has the same sign as
$f(a)$ we repeat the process on $[c,b]$
