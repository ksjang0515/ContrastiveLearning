import jax
import jax.numpy as jnp
import flax
import optax
from functools import partial


def cross_entropy(logits, labels):
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.mean(jnp.sum(logp * labels, axis=1))
    return loss

def cosine_similarity(v1, v2, eps=1e-5):
    return jnp.dot(v1, v2) / (jnp.linalg.norm(v1, ord=2)*jnp.linalg.norm(v2, ord=2) +eps)


def create_loss_n_grad(*, apply_fn, loss_fn):
    @jax.jit
    def loss_n_grad(params, batch):
        def get_loss(params, images, labels):
            y = apply_fn(params, images)
            loss = loss_fn(y, labels)
            return loss
        
        loss_grad_fn = jax.value_and_grad(get_loss)
        l, g = loss_grad_fn(params, batch['images'], batch['labels'])
        
        return l, g

    return loss_n_grad
    

