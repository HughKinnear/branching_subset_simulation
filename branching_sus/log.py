def verbose(method):
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            return method(self, *args, **kwargs)
    return wrapper


class Log:

    def __init__(self, verbose):
        self.verbose = verbose

    @verbose
    def create_level(self, level):
        print(f"Level {level.name} created.")

    @verbose
    def partition(self, set_count):
        print(f"Partition of size {set_count} found.")

    @verbose
    def terminate(self):
        print("No more options, algorithm terminated.")

    @verbose
    def constant_stop(self, level):
        print(f"Level {level.name} stopped due to constant performance.")

    @verbose
    def fail_stop(self, level):
        print(f"Level {level.name} stopped, enough samples with target performance.")






