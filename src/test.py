import data
import numpy as np
import flax
import jax
import jax.numpy as jnp
import os
from model.encoder import Resnet50

model = Resnet50(num_classes=1000)
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, shape=(1,224,224,3))
tab = flax.linen.tabulate(model, key)
print(tab(x))


