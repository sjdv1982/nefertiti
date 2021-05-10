""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import sys
import numpy as np

BASICTYPES = (dict, list, tuple, int, float, bool, str, np.number)

from numpy import uint32 as uint

default_validators = {}

ndarray = np.ndarray

class ShapeError(TypeError):
    pass

def validate_coor_dtype(arr: ndarray, state: "State") -> None:    
    from .functions.parse_pdb import atomic_dtype
    if arr.ndim != 1:
        raise ShapeError(arr.shape)
    if arr.dtype is not atomic_dtype:
        raise TypeError(arr.dtype)


def ndarray_shape_validator(arr: ndarray, state: "State", shape_expr: str) -> None:
    shape = state.eval_in_scope(shape_expr, evaluate_strings=True)
    if len(shape) != len(arr.shape):
        raise ShapeError(shape, arr.shape)
    for x, y in zip(shape, arr.shape):
        if x == -1:
            continue
        if x != y:
            raise ShapeError(shape, arr.shape)


default_validators["ndarray"] = ndarray_shape_validator

class TypedList(list):
    _type = None
    def __init__(self, value=[]):
        for v in value:
            if not isinstance(v, self._type):
                raise TypeError(v)
        super().__init__(value)

def ListOf(x):
    return type("ListOf({})".format(x), (TypedList,), {"_type": x})

class _StateScope(dict):
    def __init__(self, state: "State"):
        self.state = state
    def __getitem__(self, attr):
        try:
            return getattr(self.state, attr)
        except AttributeError:
            if attr in self.state._globals:
                return self.state._globals[attr]
            raise NameError(attr) from None

class State:
    parent = None
    _state = {}
    _globals = {}
    def __init__(self, **kwargs):
        self._globals = sys.modules[self.__module__].__dict__
        for argname, argvalue in kwargs.items():
            super().__setattr__(argname, argvalue)
        for argname, argvalue in kwargs.items():
            if argname == "parent":
                continue
            setattr(self, argname, argvalue)    

    def __setattr__(self, attr, value):
        if attr == "parent" or attr.startswith("_"):
            super().__setattr__(attr, value)
            return
        if attr not in self._state:
            raise AttributeError(attr)
        type_descr = self._state[attr]
        validator = None
        validator_args = None
        if isinstance(type_descr, str):
            typename = type_descr
        else:
            assert isinstance(type_descr, tuple), (attr, type_descr)
            typename = type_descr[0]
            if len(type_descr) == 2:
                if callable(type_descr[1]):
                    validator = type_descr[1]
                else:
                    assert typename in default_validators, typename
                    validator = default_validators[typename]
                    validator_args = type_descr[1]
            else:
                assert len(type_descr) == 3
                assert callable(type_descr[1])
                validator, validator_args = type_descr[1:]
        #typeclass = self._globals[typename]
        typeclass = self.eval_in_scope(typename, True)
        if isinstance(value, BASICTYPES):
            if issubclass(typeclass, State):
                if not isinstance(value, dict):
                    raise TypeError(attr, typeclass, type(value))
                value = typeclass(**value, parent=self)
            else:
                cast_value = typeclass(value)
                if type(value)(cast_value) != value:
                    raise TypeError(type(value), value, type(cast_value), cast_value)
                value = cast_value
        if not isinstance(value, typeclass):
            raise TypeError(attr, type(attr), typeclass)
        if validator is None:
            if typename in default_validators:
                validator = default_validators[typename]            
        if validator is not None:
            if validator_args is None:
                validator(value, self)
            else:
                validator(value, self, validator_args)
        super().__setattr__(attr, value)
        self.validate()

    def __getattr__(self, attr):
        in_state = False
        stateholder = self
        while stateholder is not None:
            if attr in stateholder._state:
                in_state = True
                break
            stateholder = stateholder.parent
        if not in_state:
            raise AttributeError(attr)
        if attr in self._state:
            return None
        parent = self.parent
        assert parent is not None  # logical consistency
        return getattr(parent, attr)

    def _eval_in_scope(self, scope, expr, evaluate_strings, *, nest):
        if isinstance(expr, str):
            if nest > 0 and not evaluate_strings:
                return expr
            result = eval(expr, scope)
            return result
        elif isinstance(expr, dict):
            result = {}
            for k, v in expr.items():
                eval_v = self._eval_in_scope(
                    scope, v, evaluate_strings, 
                    nest=nest+1
                )
                result[k] = eval_v
            return result
        elif isinstance(expr, (list, tuple)):
            result = []
            for v in expr.items():
                eval_v = self._eval_in_scope(
                    scope, v, evaluate_strings, 
                    nest=nest+1
                )
                result.append(eval_v)
            return tuple(result)
        else:
            return expr       

    def eval_in_scope(self, expr: str, evaluate_strings: bool):
        scope = _StateScope(self)
        return self._eval_in_scope(scope, expr, evaluate_strings, nest=0)

    def validate(self) -> None:
        pass

