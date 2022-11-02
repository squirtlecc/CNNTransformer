# import inspect
# import six

# def isStr(x):
#     # 检查是否为字符串
#     return isinstance(x, six.string_types)


class Register(object):
    # 注册器 用于注册维护多个模型
    def __init__(self, registry_name):
        self._name = registry_name
        self._module_dict = dict()

    def __contains__(self, key):
        return self.get(key) is not None
    
    def __repr__(self):
        # 自定义输出实例化对象的信息
        # http://c.biancheng.net/view/2367.html
        format_str = self.__class__.__name__+"(name={},items={})".format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        # property 修饰方法调用是可以省去()
        return self._name
    
    @property
    def getModuleDict(self):
        return self._module_dict
    
    def get(self, key):
        return self._module_dict.get(key, None)

    def _registerModule(self, module_class):
        """
        Register a module
        Args:
            moduld(:obj:`nn.Module`):module to be registered
        """
        # if not inspect.isclass(module_class):
        if not callable(module_class):
            raise TypeError(f"Module must be a class.but got{type(module_class)}")
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name}")
        self._module_dict[module_name] = module_class
        
   
    def registerModule(self, module_class):
        self._registerModule(module_class)

        # register must be return the origin class
        # if not return when your register some class to a instance 
        # the instance was none ,so when you return it, it's fine
        # @Register.registerModule
        # class something:
        # equals Register.registerModule(something)
        return module_class

def buildFromConfig(cfg, register, default_args=None):
    """
    Build a module from config dict.

    Args:
        cfg (dict): A yaml file include config dict. It should at least contain the key "type".
        register (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg,dict) and 'type' in cfg
    assert isinstance(default_args,dict) or default_args is None
    assert isinstance(register,Register)

    args = cfg.copy()

    obj_type = args.pop('type')
    if isinstance(obj_type,str):
        obj_name = obj_type
        obj_class = register.get(obj_name)
        if obj_class is None:
            raise KeyError(f"|x| {obj_name} is not in {register.name} registry")
    elif isinstance(obj_type,type):
    # elif inspect.isclass(obj_type):
        obj_class = obj_type
    else:
        raise typeError(f"Type must be a str or valid type, but got{type(obj_type)}")

    if default_args is not None:
        for name,value in default_args.items():
            args.setdefault(name,value)
    
    return obj_class(**args)
       