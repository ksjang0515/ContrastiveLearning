import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from tqdm import tqdm

import data
from model.encoder import Resnet50Classifier
from utils import init_model
from loss import create_update_fn, cross_entropy

SEED=0
BATCH_SIZE=64
EPOCH=1
NUM_CLASSES=1000


def main():
    key = jax.random.key(SEED)

    # dataset
    dataset = data.get_train()
    key, subkey = jax.random.split(key)
    sampler = data.RandomSampler(subkey, len(dataset), BATCH_SIZE, redundant=True)

    dataloader = data.ImagenetDataLoader(
        dataset=dataset,
        batch_sampler=sampler
    )

    # model
    model = Resnet50Classifier(num_classes=1000)
    params = init_model(model)

    # optim
    tx = optax.adam(0.001)
    opt_state = tx.init(params)

    # update_fn
    update_fn = create_update_fn(apply_fn=model.apply, loss_fn=cross_entropy, tx=tx)
    
    for _ in tqdm(range(EPOCH), desc='epoch', position=0):
        iter_bar = tqdm(dataloader, desc='iter', position=1)
        for batch in iter_bar:
            params, opt_state, loss = update_fn(params, opt_state, batch)
            iter_bar.set_postfix(dict(loss=loss))
            print("DONE, quitting")
            return



if __name__ == "__main__":
    main()

