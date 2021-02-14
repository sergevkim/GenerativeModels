class InvertImage:
    @staticmethod
    def __call__(pic):
        return 1 - pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

