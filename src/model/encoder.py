import flax
import flax.linen as nn
import jax.numpy as jnp

from typing import Sequence, Any, Callable, Tuple
from functools import partial


ModuleDef = Any

class Bottleneck(nn.Module):
    features: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, inputs):
        x, residual = inputs, inputs

        x = self.conv(self.features, (1, 1))(x)
        x = self.norm()(x)
        x = self.act(x)

        x = self.conv(self.features, (3, 3), strides=self.strides)(x)
        x = self.norm()(x)
        x = self.act(x)

        x = self.conv(self.features*4, (1, 1))(x)
        x = self.norm(scale_init=nn.initializers.zeros_init())(x)

        if residual.shape != x.shape:
            residual = self.conv(self.features*4, (1, 1), self.strides)(residual)
            residual = self.norm()(residual)

        x = x + residual
        x = self.act(x)
        return x

class Resnet(nn.Module):
    stage_sizes: Sequence[int]
    num_classes: int
    dtype: Any = jnp.float32
    act: Callable = nn.activation.relu
    conv: ModuleDef = nn.Conv
    num_filters: int = 64

    @nn.compact
    def __call__(self, inputs, train:bool=False):
        x = inputs
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)])(x)
        x = norm()(x)
        x = self.act(x)

        x = nn.max_pool(x, (3, 3), (2, 2), 'SAME')
        for i, block_num in enumerate(self.stage_sizes):
            for j in range(block_num):
                strides = (2, 2) if i>0 and j==0 else (1, 1)
                x = Bottleneck(self.num_filters*2**i, strides=strides, conv=conv, norm=norm, act=self.act)(x)
        
        x = jnp.mean(x, (1, 2), dtype=self.dtype)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x

Resnet50 = partial(Resnet, stage_sizes=[3, 4, 6, 3])


class Resnet50Classifier(nn.Module):
    num_classes:int
    dtype: Any = jnp.float32
    act: Callable = nn.activation.relu
    conv: ModuleDef = nn.Conv
    num_filters: int = 64

    @nn.compact
    def __call__(self, inputs, train:bool=False):
        x = inputs

        # resnet
        x = Resnet50(
            num_classes=self.num_classes,
            dtype=self.dtype,
            act=self.act,
            conv=self.conv,
            num_filters=self.num_filters
        )(x)

        # softmax
        x = nn.activation.softmax(x)

        return x

























