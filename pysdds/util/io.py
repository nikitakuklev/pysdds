from io import BufferedReader, RawIOBase


class SizedBufferedReader(BufferedReader):
    def __init__(self, raw: RawIOBase, size):
        super().__init__(raw)
        self.size = size

    def __len__(self):
        return self.size
