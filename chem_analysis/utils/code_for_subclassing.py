
def get_subclasses(cls, depth: int | None = 0, _count: int = 0) -> set[type]:
    subs = set(cls.__subclasses__())
    if _count < depth or depth is None:
        return subs.union(*(get_subclasses(i, depth, _count) for i in subs))

    return subs


class MixinSubClassList:
    @classmethod
    def processing_classes(cls) -> set[type]:
        return get_subclasses(cls, depth=0)

    @classmethod
    def processing_algorithms(cls) -> set[type]:
        return get_subclasses(cls, depth=None)
