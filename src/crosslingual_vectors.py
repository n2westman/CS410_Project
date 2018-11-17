import torch
import logging
import os
import io
import array
import six
from tqdm import tqdm
from torchtext.vocab import Vectors

logger = logging.getLogger("data")

class Crosslingual(Vectors):

    def __init__(self, name, language='en', **kwargs):
        self.name = name
        self.language = language
        super(Crosslingual, self).__init__(self.name, **kwargs)

    def cache(self, name, cache=None, url=None, language='en'):
        path = name
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(name))

        # str call is necessary for Python 2/3 compatibility, since
        # argument must be Python 2 str (Python 3 bytes) or
        # Python 3 str (Python 2 unicode)
        itos, vectors, dim = [], array.array(str('d')), None

        # Try to read the whole file with utf-8 encoding.
        binary_lines = False
        try:
            with io.open(path, encoding="utf8") as f:
                lines = [line for line in f]

        # If there are malformed lines, read in binary mode
        # and manually decode each word from utf-8
        except:
            logger.warning("Could not read {} as UTF8 file, "
                           "reading file as bytes and skipping "
                           "words with malformed UTF8.".format(path))
            with open(path, 'rb') as f:
                lines = [line for line in f]
            binary_lines = True

        logger.info("Loading vectors from {}".format(path))
        lines = filter(lambda x: x.split(':')[0] == self.language, lines)
        for line in tqdm(lines, unit_scale=True, miniters=1, desc=name):
            # Explicitly splitting on " " is important, so we don't
            # get rid of Unicode non-breaking spaces in the vectors.
            entries = line.rstrip().split(b" " if binary_lines else " ")

            word, entries = entries[0].split(':')[-1], entries[1:]
            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif len(entries) == 1:
                logger.warning("Skipping token {} with 1-dimensional "
                               "vector {}; likely a header".format(word, entries))
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(word, len(entries), dim))

            if binary_lines:
                try:
                    if isinstance(word, six.binary_type):
                        word = word.decode('utf-8')
                except:
                    logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                    continue
            vectors.extend(float(x) for x in entries)
            itos.append(word)

        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.Tensor(vectors).view(-1, dim)
        self.dim = dim