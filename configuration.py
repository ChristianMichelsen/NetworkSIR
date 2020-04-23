import yaml

class DotDict(dict):
    """
    Class that allows a dict to indexed using dot-notation.
    Example:
    >>> dotdict = DotDict({'first_name': 'Christian', 'last_name': 'Michelsen'})
    >>> dotdict.last_name
    'Michelsen'
    """

    def __getattr__(self, item):
        if item in self:
            return self.get(item)
        raise KeyError(f"'{item}' not in dict")

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
            return
        raise KeyError(
            "Only allowed to change existing keys with dot notation. Use brackets instead."
        )


def dict_to_dotdict(dict):
    dot_dict = DotDict()
    for key, val in dict.items():
        if isinstance(val, str) and val.lower() == "none":
            val = None
        dot_dict[key] = val
    return dot_dict


def load():
    with open("cfg.yaml", "r") as file:
        cfg = dict_to_dotdict(yaml.safe_load(file))
    return cfg
