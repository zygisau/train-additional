class holder:
    pass


def dotted(object):

    if isinstance(object, dict):
        parent = holder()

        for key in object:

            child = object.get(key)
            setattr(parent, key, dotted(child))

        return parent

    return object
