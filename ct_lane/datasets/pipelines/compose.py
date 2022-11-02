import collections

from utils.register import buildFromConfig

from ..builder import PIPELINES

class ComposePipeline(object):
    """Compose multiple transform sequentially.
    Args:
        transforms (Sequence[dict | callable]): Sequence of process object or
            config dict to be composed.
    """


    def __init__(self, transforms, cfg):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = buildFromConfig(transform, PIPELINES, default_args=dict(cfg=cfg))
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transform sequentially.
        Args:
            data (dict): A result dict contains the data to process.
        Returns:
           dict: Transform data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string