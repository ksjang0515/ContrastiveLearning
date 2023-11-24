import hashlib

import flax
import jax
import jax.numpy as jnp


def init_model(model):
    key = jax.random.key(0)
    params = model.init(key, jnp.ones((1, 224, 224, 3), jnp.float32))
    return params

def fold_in_name(rng, name):
    # use name(string) instead of number for creating new rng stream
    # simple version of flax.core.scope._fold_in_static

    m = hashlib.sha1()
    m.update(name.encode('utf-8'))
    d = m.digest()
    hash_int = int.from_bytes(d[:4], byteorder='big')

    return jax.random.fold_in(rng, jnp.uint32(hash_int))
