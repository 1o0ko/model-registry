
# Model registry and configuration example
## Introduction

In this short example we will look at how allen-nlp performs class registration. The overal goal of this proposal is to introduce new method of performing experiments:

```bash
> cat params.json
```
```json
{
    "name": "sequential",
    "params": {
        "modules": {
            "Embedder": {
                "name": "lookup",
                "params": {
                    "vocab_size": 100,
                    "dim": 5
                }
            },
            "Encoder": {
                "name": "mp-projection",
                "param": {
                    "input_dim": 5
                }
            }
        }
    }
}
```

And then in code:
```python
with open('params.json') as f:
    params = Params(json.loads(f))

name = params.pop('name')
model_params = params.pop('params')

model = Model.by_name(name).from_params(model_params)
```

## Why?
It's way easier to version the experiments that way. 

Additionally, this is the first step towards systematic HP - search. Having defined model in a json file, we open the door for a HP search DLS. 

```python
# Grid search
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# Define distributions for sampling:
{
    'C': scipy.stats.expon(scale=100), 
    'gamma': scipy.stats.expon(scale=.1),
    'kernel': ['rbf'], 
    'class_weight': ['balanced', None]
}
```

In the next section we will walk through simplified [allennp](https://github.com/allenai/allennlp) code to see an example implementation of this functionality. 

# Registration

## Registry


```python
import logging

from collections import defaultdict
from typing import Type, TypeVar, Dict, List, Optional, Union

T = TypeVar('T')


class RegistrationError(Exception):
    def __init__(self, message):
        super(RegistrationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class Registrable:
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: Optional[str] = None
        
    @classmethod
    def register(cls: Type[T], name: str):
        ''' Register subclass'''
        registry = Registrable._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]) -> Type[T]:
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                        name, cls.__name__, registry[name].__name__)
                raise RegistrationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry
    
    @classmethod
    def list_available(cls: Type[T]) -> List[str]:
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation
        
        if default is None:
            return keys
        elif default not in keys:
            raise RegistrationError(f"Default implementation {default} is not registered")
        else:
            [default] + [k for k in keys if k != default]
    
    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
        logger.info(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise RegistrationError("%s is not a registered name for %s" % (name, cls.__name__))
        
        return Registrable._registry[cls].get(name)
```

## Params


```python
import inspect

from collections import MutableMapping
from typing import Any
```


```python
PRIMITIVES = set([int, str, bool, float])


class Params(MutableMapping):
    ''' A recursive datastructure for parameters '''
    def __init__(self, 
                 params: Dict[str, Any]) -> None:
        self.params = params

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, value):
        ''' Ensures that dictionaries are transform to Params'''
        if isinstance(value, dict):
            return Params(value)
        if isinstance(value, list):
            value = [self._check_is_dict(v) for v in value]
        return value

    def __repr__(self):
        return str(self.__dict__)

def remove_optional(annotation: type):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        return annotation

def create_kwargs(cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    # Get the signature of the constructor.
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}
        
    # Iterate over all the constructor parameters and their annotations.
    for name, param in signature.parameters.items():
        if name == "self":
            continue
            
        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])

        # The parameter is optional if its default value is not the "no default" sentinel.
        default = param.default
        optional = default != inspect.Parameter.empty
        
        # Some constructors expect a non-parameter items, e.g. tokens: TokenDict
        if name in extras:
            kwargs[name] = extras[name]
        elif annotation in PRIMITIVES:
            kwargs[name] = annotation((
                params.pop(name, default)
                if optional 
                else params.pop(name)))
        else:
            raise RegistrationError("Could not create object from config")
        
        # TODO: explain what to do when the parameter type is itself constructible from_params
        
    return kwargs

    
class FromParams:
    ''' Mixin to add ``from_params`` method to mixes classes '''

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        logger = logging.getLogger(__name__)
        logger.info(
            f"instantiating class {cls} from params {getattr(params, 'params', params)} "
            f"and extras {extras}"
        )
        if params is None:
            return None
        
        if cls.__init__ == object.__init__:
            # This class does not have an explicit constructor, so don't give it any kwargs.
            # Without this logic, create_kwargs will look at object.__init__ and see that
            # it takes *args and **kwargs and look for those.
            kwargs: Dict[str, Any] = {}
        else:
            # This class has a constructor, so create kwargs for it.
            kwargs = create_kwargs(cls, params, **extras)
            print(kwargs)

        return cls(**kwargs)  # type: ignore
        
```

# Framework mock
In this section we introduce a mock framework with forawrd pass only

## Building blocks


```python
import sys
import numpy as np

from typing import Any
from abc import ABC, abstractmethod


class Module(ABC):
    def __call__(self, inputs: Any) -> np.ndarray:
        return self.forward(inputs)
    
    @abstractmethod
    def forward(self, inputs: Any) -> np.ndarray:
        pass

class Model(Registrable, Module):
    pass

@Model.register("sequential")
class Sequential(Module):
    def __init__(self, modules: List[Module]) -> None:
        self.modules = modules
    
    def forward(self, inputs: Any) -> np.ndarray:
        for m in self.modules:
            inputs = m(inputs)
        
        return inputs

    
    @classmethod
    def from_params(self, params: Params) -> Module:
        modules = []
        for cls_name, cls_params in params['modules'].items(): 
            cls = getattr(sys.modules[__name__], cls_name).by_name(
                cls_params['name']
            )
            modules.append(cls.from_params(
                cls_params['params']
            ))
        
        return Sequential(modules)
```

## Embedders


```python
import numpy as np

class Embedder(Module, Registrable):
    def forward(self, inputs: List[int]) -> np.ndarray:
        pass

@Embedder.register("random")
class Random(Embedder, FromParams):
    def __init__(self, low: float, high: float, dim: int = 5):
        self.low: float = low
        self.high: float = high
        self.dim = dim
    
    def forward(self, inputs: List[int]) -> np.ndarray:
        ''' Returns random embedding '''
        return np.random.uniform(
            low=self.low, high = self.high, size=(len(inputs), self.dim)
        )

@Embedder.register("lookup")
class LookupTable(Embedder, FromParams):
    def __init__(self, vocab_size: int, dim: int = 5):
        self.embeddings: Dict[int, np.ndarray] = {
            idx : np.random.randn(dim) for idx in range(1, dim + 1)
        }
        self.embeddings[0] = np.zeros(dim)
    
    def forward(self, inputs: List[int]) -> np.ndarray:
        ''' Returns values form lookup table '''
        return np.vstack([
            self.embeddings[idx] if idx in self.embeddings else self.embeddings[0]
            for idx in inputs
        ])
```

## Encoders


```python
class Encoder(Module, Registrable):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    
@Encoder.register("mp-projection")
class MeanPoolingProjection(Encoder, FromParams):
    def __init__(self, input_dim: int, output_dim: int = 3) -> None:
        self.projection_matrix = np.random.uniform(size=(input_dim, output_dim))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.projection_matrix
```

## Model Created Manually


```python
model = Sequential([
    LookupTable(10, 5),
    MeanPoolingProjection(5, 3)
])
```


```python
model([1, 2, 12])
```




    array([[ 2.52787215,  0.73180424,  2.85418015],
           [-0.25555594, -1.2499665 , -0.73276124],
           [ 0.        ,  0.        ,  0.        ]])



## Model Created from Config


```python
params = Params({
    'name': 'sequential',
    'params': {
        'modules': {
            'Embedder': {
                'name': 'lookup',
                'params': {
                    'vocab_size': '100',
                    'dim': '5'
                }
            },
            'Encoder': {
                'name': 'mp-projection',
                'params': {
                    'input_dim': 5
                }
            }
        }
    }
})
```


```python
name = params.pop('name')
model_params = params.pop('params')
```


```python
model = Model.by_name(name).from_params(model_params)
```

    {'vocab_size': 100, 'dim': 5}
    {'input_dim': 5, 'output_dim': 3}



```python
model([1,2, 12])
```




    array([[0.58250073, 1.10438449, 0.35785939],
           [1.1865377 , 0.41630972, 1.05526072],
           [0.        , 0.        , 0.        ]])

