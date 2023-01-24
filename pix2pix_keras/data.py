from .utils import tf, np


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, read_func, batch_size=16, train_test_split: float=1.):
        self.images = np.array(images)
        self.batch_size = batch_size
        
        np.random.seed(42)
        np.random.shuffle(self.images)

        split_size = int(len(self.images)*train_test_split)
        self.images = self.images[:split_size] if train_test_split > 0. else self.images[split_size:]
        print(len(self.images), 'train' if train_test_split > 0. else 'test', 'images found')

        self.read_func = read_func

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images[idx*self.batch_size:idx*self.batch_size+self.batch_size]

        xs = []
        ys = []
        for image in batch_images:
            x, y = self.read_func(image)
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)

        return xs, ys

