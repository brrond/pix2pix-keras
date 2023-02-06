from .utils import tf, np


class DataGenerator(tf.keras.utils.Sequence):
    """
    An implementation of DataGenerator that allows automatic data generator creation.
    Current generator takes images keys and read function as input for data processing.
    
    Each iteration generator calls read_func with specific key, like:
    for key in images:
        x, y = read_func(key)
    where x & y - input and output images respectively.
    
    Current class is inherited from keras.utils.Sequence.

    Attributes
    ----------
    images : list
        a list of image's keys (could be file names, integers etc.).
    batch_size : int
        a size of the batch for one iteration of generation (default 16).
    read_func : function
        image reading function.

    Methods
    -------
    __len__()
        Returns number of batches.

    __getitem__(idx)
        Returns x, y pair with shapes (batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    """

    def __init__(self, images: list, read_func, batch_size: int=16, train_test_split: float=1., random_state=None):
        """
        Parameters
        ----------
        images : int 
            Image's keys.
        read_func : function
            The user specified read_function.
        batch_size : int
            The number of images per __getitem__ call (default 16).
        train_test_split : float
            Specifies train/test -ing split. 
            If value greater than zero, then train split of size len(images)*train_test_split selected.
            If value less than zero, then test split of size len(images)*train_test_split is selected.
        random_state : int, None
            Random state seed for data shuffling.
            If None (default) no seed will be used. See np.random for more details on this one.
        """

        self.images = np.array(images)  # images will be saved as numpy array to allow numpy operations
        self.batch_size = batch_size
        
        if random_state is not None:
            np.random.seed(random_state)  # random state seed for easy-to-reproduce tests
        np.random.shuffle(self.images)  # shuffle keys

        split_size = int(len(self.images)*train_test_split)  # acquire split size
        self.images = self.images[:split_size] if train_test_split > 0. else self.images[split_size:]  # get actual split
        print(len(self.images), 'train' if train_test_split > 0. else 'test', 'images found')  # inform user about split
        # TODO: Add verbose for this print

        self.read_func = read_func

    def __len__(self):
        """
        Returns
        -------
        int
            a length of generator (count of batches).
        """
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int 
            Number of generation batch.

        Returns
        -------
        numpy.array, numpy.array
            that represent x and y data.
            Shapes of both array are (batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CH)
        """

        batch_images = self.images[idx*self.batch_size:idx*self.batch_size+self.batch_size]  # get batch keys

        xs = []
        ys = []
        for image in batch_images:
            x, y = self.read_func(image)
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)

        return xs, ys  # returns 2 * (batch_size, IMAGE_H, IMAGE_W, IMAGE_CH)

