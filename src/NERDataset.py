from torchtext import data


class NERDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, separator="\t", encoding='utf8', **kwargs):
        examples = []
        columns = []
        count = 0
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        columns.append([count])
                        count+=1
                        example = data.Example.fromlist(columns, fields)
                        assert len(example.inputs_word) == len(example.labels)
                        examples.append(example)
                    columns = []
                elif('-DOCSTART-' not in line):
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)
            if columns:
                columns.append([count])
                example = data.Example.fromlist(columns, fields)
                assert len(example.inputs_word) == len(example.labels)
                examples.append(example)
        super(NERDataset, self).__init__(examples, fields, **kwargs)
