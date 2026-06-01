"""Typing for GradaBeam"""

from typing import Union, Any

SequenceType = str
SamplesType = list[SequenceType]
TISMType = list[dict[str, float]]

# Simple base classes for type annotations
class ModelClass:
    pass

class TISMModelClass(ModelClass):
    pass

class PyTorchDifferentiableModel(ModelClass):
    pass

ModelType = Union[ModelClass, Any]
