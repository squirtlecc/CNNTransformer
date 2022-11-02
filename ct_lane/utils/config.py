# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import copy


BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']


def check_file_exist(filename: str, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class ConfigDict(dict):

    # Start-- fixed copy issues from
    # https://github.com/mewwts/addict/blob/master/addict/addict.py
    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)

        return item

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                (not isinstance(self[k], dict)) or
                    (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def copy(self):
        return copy.copy(self)
    # --End

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getitem__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

    def haskey(self, name) -> bool:
        if name in self.keys():
            return True
        return False

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

class Config:
    """A facility for config and config files.
    It supports common file formats as configs: json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.yaml')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.yaml"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)

        if filename.endswith(('.yml', '.yaml')):
            import yaml
            cfg_dict = yaml.safe_load(open(filename).read())
        elif filename.endswith(('.json')):
            import json
            cfg_dict = json.load(open(filename))
        else:
            raise IOError('Only yml/yaml/json type are supported now!')

        cfg_text = ''
        with open(filename, 'r') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict `a` into dict `b` (non-inplace). values in `a` will
        # overwrite `b`.
        # copy first to avoid inplace modification
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from base '
                        f'because {k} is a dict in the child config but is of '
                        f'type {type(b[k])} in base config. You may set '
                        f'`{DELETE_KEY}=True` to ignore the base config')
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename: str):
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)  

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def text(self):
        return self._text
    
    @property
    def cfg_dict(self):
        return self._cfg_dict

    # get value though dot func
    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    # get value though map func
    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    # set value though dot func
    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    # for next()
    def __iter__(self):
        return iter(self._cfg_dict)

    def haskey(self, name) -> bool:
        return self._cfg_dict.haskey(name)

    def isExists(self, name: str) -> bool:
        names = name.split('.') if name.split('.') is not None else [name]
        _temp_dict = self._cfg_dict.copy()
        check_keys = _temp_dict.keys()

        for n in names:
            if n in check_keys:
                _temp_dict = _temp_dict[n]
                if isinstance(_temp_dict, dict):
                    check_keys = _temp_dict.keys()
                else:
                    if names[-1] != n:
                        return False
                continue
            return False
        return True

    # def addNone(self, name: str) -> None:
    #     def deepSearch(d1,d2):
    #         for k in d2.keys():
    #             if k not in d1.keys():
    #                 d1[k] = d2[k]
    #             else:
    #                 deepSearch(d1[k],d2[k])

    #     names = name.split('.') if name.split('.') is not None else [name]
    #     names.reverse()
    #     inject = None
    #     for n in names:
    #         inject = {n:inject}
    #     deepSearch(self._cfg_dict, inject)


if __name__ == "__main__":
    print(Config.fromfile('./configs/tusimple.yml'))