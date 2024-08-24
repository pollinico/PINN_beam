import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np
from jax.nn import tanh, relu
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

# Network initialization
def random_layer_params(m, n, key, scale):
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key, (m, n)), jnp.zeros(n)

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, 2.0/(jnp.sqrt(m+n))) \
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Neural Network 
@jit
def predict(params, X):
    A =  X
    for w, b in params[:-1]:
        A = tanh(jnp.dot(A, w) + b)
    final_w, final_b = params[-1]
    A = jnp.dot(A, final_w) + final_b
    return A

# Derivatives of u(x)
@jit
def net_u(params, x):
    u_pred = predict(params, x)
    return u_pred

@jit
def net_u_grad(params, x):
    u_pred = predict(params, x)
    return u_pred[0,0]

def net_fx(params):
    def u_x(x):
        ux = grad(net_u_grad, argnums=1)(params, x) 
        return ux
    return jit(u_x)

def net_fxx(params):
    def u_xx(x):
        u_x = net_fx(params)
        uxx = grad(u_x, argnums=0)(x) 
        return uxx
    return jit(u_xx)

def net_fxxx(params):
    def u_xxx(x):
        u_xx = net_fxx(params)
        uxxx = grad(u_xx, argnums=0)(x) 
        return uxxx
    return jit(u_xxx)

def net_fxxxx(params):
    def u_xxxx(x):
        u_xxx = net_fxxx(params)
        uxxxx = grad(u_xxx, argnums=0)(x) 
        return uxxxx
    return u_xxxx

# Loss (error) function due to boundary conditions
@jit
def loss_data(params, x_BC, u_BC, uxx_BC):
    u = vmap(net_u, (None, 0))(params, x_BC)
    u_xx = net_fxx(params)
    uxx = vmap(u_xx)(x_BC.flatten())
    loss = jnp.mean((u - u_BC)**2) + jnp.mean((uxx - uxx_BC)**2)
    return loss

# Loss (error) function due to PDE residual
@jit
def loss_pde(params, x_PDE, p, EI):
    u_xxxx = net_fxxxx(params)
    uxxxx = vmap(u_xxxx)(x_PDE.flatten())
    residual = uxxxx - p/EI
    loss_pde = jnp.mean((residual.flatten())**2)
    return loss_pde

def loss_tot(params, x_BC, u_BC, uxx_BC, x_PDE, p, EI):
    loss_p = loss_pde(params, x_PDE, p, EI)
    loss_bc = loss_data(params, x_BC, u_BC, uxx_BC)
    return loss_p + loss_bc

# Network parameter update
@jit
def step(istep, opt_state, x_BC, u_BC, uxx_BC, x_PDE, p, EI):
    param = get_params(opt_state) 
    g = grad(loss_tot, argnums=0)(param, x_BC, u_BC, uxx_BC, x_PDE, p, EI)
    return opt_update(istep, g, opt_state)

# Reference analytical solution
def u_elasticLine(x, p, L, EI):
    u = p*x*(x**3 - 2*L*x**2 + L**3)/24/EI
    return u

# Main driver
if __name__ == "__main__":
    EI = 1.0 # Bending stiffness
    pLoad = 1.0 # Distributed load
    N_u = 20 + 1
    Lbeam = 1.0 # m
    dx = Lbeam / (N_u-1)
    print("dx: ", dx)
    layers = [1, 10, 1]
    Nmax = 5000 # Max number of iterations for training the NN


    params = init_network_params(layers, random.PRNGKey(1234))
    opt_init, opt_update, get_params = optimizers.adam(1e-2)
    opt_state = opt_init(params)

    x_tot   = jnp.reshape(jnp.arange(0.0, Lbeam+dx, dx), (-1,1))
    x_BC    = jnp.reshape(jnp.array([0.0, Lbeam]), (-1,1))
    u_BC    = jnp.reshape(jnp.array([0.0, 0.0]), (-1,1)) # BC on displacement u(x)
    uxx_BC  = jnp.reshape(jnp.array([0.0, 0.0]), (-1,1)) # BC on moment M(x)
    x_PDE = jnp.reshape(jnp.arange(dx, Lbeam-dx, dx), (-1,1))

    nIter = 5000 + 1
    ld_list = []
    lp_list = []

    for it in range(nIter):
        opt_state = step(it, opt_state, x_BC, u_BC, uxx_BC, x_PDE, pLoad, EI)
        if it % 1 == 0:
            params = get_params(opt_state)
            l_d = loss_data(params, x_BC, u_BC, uxx_BC)
            l_p = loss_pde(params, x_PDE, pLoad, EI)
            print(f"Loss data: {l_d}, Loss PDE: {l_p}")
            ld_list.append(l_d)
            lp_list.append(l_p)

    params = get_params(opt_state)

    plt.figure(figsize=(8, 4))
    plt.plot(ld_list, linewidth=2)
    plt.plot(lp_list, linewidth=2)
    plt.yscale("log")
    plt.legend({"BC", "PDE"})
    plt.savefig("losses.png")

    plt.figure(figsize=(8, 4))
    plt.plot( np.sort(x_tot,axis=0), -u_elasticLine(np.sort(x_tot,axis=0), pLoad, Lbeam, EI), label="exact solution", linewidth=2, linestyle="dashed")
    plt.plot( np.sort(x_tot,axis=0), -net_u(params, np.sort(x_tot,axis=0)), label="neural network", linewidth=2)
    plt.legend()
    plt.savefig("deformed_shape.png")
    plt.show()
