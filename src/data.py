from torch.utils.data import Dataset, DataLoader, Sampler
import jax
import jax.numpy as jnp
from PIL import Image
import dm_pix as pix
import os
import math

from utils import fold_in_name


DATASET_PATH = '/mnt/hdd/datasets/imagenet-1k'
TRAIN_PATH = f'{DATASET_PATH}/train'
VAL_PATH = f'{DATASET_PATH}/test'
MAPPING_FILE = f'{DATASET_PATH}/ILSVRC2012_mapping.txt'
VAL_LABEL_PATH = f'{DATASET_PATH}/ILSVRC2012_validation_ground.txt'


def read_image(image_path):
    """reads single image and return jax array, applies BiCubic resize to (256, 256) then center crop to (256, 256) as common approach"""
    image = Image.open(image_path)
    # convert non-RGB image into RGB (ex. grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    jax_arr = jnp.array(image, dtype=jnp.float32)
    jax_arr = jax.image.resize(jax_arr, (256, 256, 3), method=jax.image.ResizeMethod.CUBIC)
    jax_arr = jax_arr / 256
    return jax_arr


class ImagenetDataset(Dataset):
    def __init__(self, image_paths, labels):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = read_image(image_path)

        return image, label

def jax_collate_fn(batch):
    images = jnp.stack([b[0] for b in batch])
    labels = jnp.stack([b[1] for b in batch])
    one_hot = jax.nn.one_hot(labels, 1000)
    
    b_ = dict(images=images, labels=one_hot)
    return b_

class ImagenetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=None, persistent_workers=False, pin_memory_device=''):
        super().__init__(dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=jax_collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
                multiprocessing_context=multiprocessing_context,
                generator=generator,
                prefetch_factor=prefetch_factor,
                persistent_workers=False,
                pin_memory_device=pin_memory_device)


class RandomSampler(Sampler):
    def __init__(self, prng_key, length, batch_size, redundant:bool=False, drop_last=False):
        """Jax version of pytorch BatchSampler
        redundant - elements can be sampled more than once, if True drop_last will be ignored (iteration will be Ceilinged)
        drop_last - when redundant is False, drop_last determines whether short/insufficient batch will be dropped/filled
        """
        self._prng_key = fold_in_name(prng_key, 'RandomSampler')
        self.length = math.ceil(length/batch_size)
        self.batch_size = batch_size
        self.redundant = redundant
        self.drop_last = drop_last

    @property
    def prng_key(self):
        # this design will not work in parellel
        key, subkey = jax.random.split(self._prng_key)
        self._prng_key = key

        return subkey

    def __len__(self):
        return self.length

    def __iter__(self):
        if self.redundant:
            num_iter = math.ceil(self.length / self.batch_size)
            
            for _ in range(num_iter):
                index = jax.random.randint(self.prng_key, (self.batch_size, ), 0, self.length)
                yield index
            
            return
        
        index_list = jax.random.permutation(self.prng_key, jnp.arange(self.length))
        if self.drop_last:
            num_iter = self.length // self.batch_size
            index_list = index_list[:num_iter * self.batch_size]

        else:
            num_iter = math.ceil(self.length / self.batch_size)
            if self.length % self.batch_size:
                remain = self.batch_size - self.length % self.batch_size 
                index_list = jnp.concatenate((index_list, index_list[:remain]))

        index_list = jnp.reshape(index_list, (num_iter, self.batch_size))
        yield from index_list


def get_dataset(image_paths, labels):
    """returns torch.utils.data.Dataset instance"""

    return ImagenetDataset(image_paths, labels)

def get_train():
    # create dictionary of id to index
    id_to_idx = {}
    with open(MAPPING_FILE, 'r') as f:
        for line in f:
            idx, id = line.strip().split(' ')
            id_to_idx[id] = idx

    # get image paths
    image_paths = []
    labels = []
    for img_name in os.listdir(TRAIN_PATH):
        if not img_name.endswith(".JPEG"):
            pass

        img_id = img_name.split('_')[0]
        if img_idx:=id_to_idx.get(img_id):
            img_path = f'{TRAIN_PATH}/{img_name}'
            image_paths.append(img_path)
            labels.append(int(img_idx))

    return get_dataset(image_paths, labels)

def get_val():
    image_names = os.listdir(VAL_PATH)
    image_names.sort()
    image_paths = list(map(lambda x: f'{VAL_PATH}/{x}', image_names))
    
    labels = []
    with open(VAL_LABEL_PATH, 'r') as f:
        for line in f:
            idx = int(line.strip())
            labels.append(idx)

    return get_dataset(image_paths, labels)



