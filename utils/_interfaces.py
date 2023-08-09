

class DisposableModel:

    def unload_model(self):
        raise NotImplementedError()

    def dispose(self):
        self.unload_model()

    def __del__(self):
        self.dispose()
