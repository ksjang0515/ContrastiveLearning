import inspect

import flax
import jax
import jax.numpy as jnp
import dm_pix as pix


class Compose:
    def __init__(self, prng_key, augments):
        self._prng_key = prng_key
        self.augments = augments


    @property
    def prng_key(self):
        key, subkey = jax.random.split(self._prng_key)
        self._prng_key = key

        return sub

    @jax.jit
    def __call__(self, x):
        for aug in self.augments:
            aug_map = jnp.vmap(aug)
            x = aug_map(x, prng_key=self.prng_key)

        return x

def random_apply(aug, prob):
    def random_aug(x, prng_key, **kwargs):
        should_transform = jax.random.bernoulli(prng_key, prob)
        result = jax.lax.cond(should_transform, aug, lambda x: x, x)

        return result

def prng_key_arg_wrapper(aug):
    params = inspect.signature(aug).parameters
    prng_key_exists = [p[0] == 'prng_key' for p in params]
    if any(prng_key_exists):
        return aug

    def wrapper_func(x, prng_key):
        return aug(x)
    return wrapper_func
