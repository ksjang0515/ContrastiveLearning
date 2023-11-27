import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
from tqdm import tqdm


import data
from model.encoder import Resnet50Classifier
from utils import init_model
from loss import create_loss_n_grad, cross_entropy

SEED=0
BATCH_SIZE=32
EPOCH=50
NUM_CLASSES=1000
CHKPT_PATH = '/home/kyusang/dev/dl/flax/ContrastiveLearning/checkpoint/resnet50'
MAX_TO_KEEP = 30
CHKPT_ITER = 3000


def get_chkpt_manager(chkpt_path, max_to_keep):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    chkpt_manager = orbax.checkpoint.CheckpointManager(chkpt_path, checkpointer, options)

    return chkpt_manager

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

    # chkpt
    chkpt_manager = get_chkpt_manager(CHKPT_PATH, MAX_TO_KEEP)

    # model
    model = Resnet50Classifier(num_classes=1000)
    params = init_model(model)

    # optim
    tx = optax.adam(0.001)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    # loss_n_grad
    loss_n_grad = create_loss_n_grad(apply_fn=state.apply_fn, loss_fn=cross_entropy)
    
    cnt = 0
    chkpt_cnt = 0
    for _ in tqdm(range(EPOCH), desc='epoch', position=0):
        iter_bar = tqdm(dataloader, desc='iter', position=1)
        for batch in iter_bar:
            loss, grad = loss_n_grad(state.params, batch)
            state = state.apply_gradients(grads=grad)
            iter_bar.set_postfix(dict(loss=loss))

            with open('loss_history.txt', 'a') as f:
                f.write(f'{loss}\n')

            cnt += 1
            if cnt % CHKPT_ITER == 0:
                chkpt = {'model': state}
                save_args = orbax_utils.save_args_from_target(chkpt)
                chkpt_manager.save(chkpt_cnt, chkpt, save_kwargs={'save_args':save_args})
                chkpt_cnt += 1
                cnt = 0


if __name__ == "__main__":
    main()

