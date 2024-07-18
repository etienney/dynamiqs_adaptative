# dynamiqs_adaptative - Inequalities ab initio
A version of dynamiqs implementing a solver where the object can be truncated in modes according to some inequalities. It allows for computing the Lindblad equation according to the selected modes only.

## Exemple
```
import dynamiqs_adaptative as dq
import jax.numpy as jnp

def jump_ops2(n_a,n_b):
    kappa=1.0
    a_a = dq.destroy(n_a)
    identity_a = jnp.identity(n_a)
    a_b = dq.destroy(n_b)
    identity_b = jnp.identity(n_b)
    tensorial_a_b = dq.tensor(identity_a,a_b)
    jump_ops = [jnp.sqrt(kappa)*tensorial_a_b]
    return jump_ops
def H2(n_a,n_b, alpha = 0):
    a_a = dq.destroy(n_a)
    identity_a=jnp.identity(n_a)
    a_b = dq.destroy(n_b)
    identity_b=jnp.identity(n_b)
    tensorial_a_a=dq.tensor(a_a,identity_b)
    tensorial_a_b=dq.tensor(identity_a,a_b)
    H=(
        (tensorial_a_a@tensorial_a_a - jnp.identity(n_a*n_b)*alpha**2) @ 
        dq.dag(tensorial_a_b) + 
        dq.dag(tensorial_a_a@tensorial_a_a - jnp.identity(n_a*n_b)*alpha**2) @ 
        tensorial_a_b - 
        tensorial_a_b - dq.dag(tensorial_a_b)
    )
    return H
def rho2(n_a,n_b):
    return dq.fock_dm((n_a,n_b),(0,0)) #|00><00|

ntemps2=1.0
steps2=64
t_span2=jnp.linspace(0,ntemps2,steps2)
n_a=65
n_b=65
tensorisation=(n_a, n_b)

def ineq(a, b):
    return a+b
def ineq2(a, b):
    return 3*a+b
inequalities = [[ineq, 15], [ineq2, 13]]
res2 = dq.mesolve(H2(n_a, n_b, 1.0), jump_ops2(n_a, n_b), rho2(n_a, n_b), t_span2, 
    solver=dq.solver.Tsit5(), gradient=dq.gradient.Autograd(), 
    options = dq.Options(tensorisation=tensorisation, inequalities=inequalities)
)
print(res2.states)
print(res2)
```

# Installation 

```shell
pip install git+https://github.com/etienney/dynamiqs_adaptative.git@inequalities_ab_initio
```
