import itertools
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Parameter(ABC):
    key: str

    @abstractmethod
    def check(self, value):
        raise NotImplementedError


@dataclass
class StringParameter(Parameter):
    choices: Optional[List[str]] = None

    def check(self, value):
        if self.choices is not None and value not in self.choices:
            # FIXME: Add warning text
            raise RuntimeError


@dataclass
class FloatParameter(Parameter):
    min: Optional[float] = None
    max: Optional[float] = None

    def check(self, value):
        if self.min is not None and value < self.min:
            # FIXME: Add warning text
            raise RuntimeError

        if self.max is not None and self.max < value:
            # FIXME: Add warning text
            raise RuntimeError


class DispModel(ABC):
    PARAMS = []

    # I am using class methods here, because it seems like you don't want to save
    # information about the scenario in an instance
    @classmethod
    def _check_params(cls, **kwds):
        for param in cls.PARAMS:
            param.check(kwds[param.key])

    @classmethod
    def _vectorize_inputs(cls, **kwds):
        # Could use pandas.MultiIndex.from_product to make these too

        # Combine all of the scenarios
        scenarios = itertools.product(
            *[
                [value] if not isinstance(value, (list, np.ndarray)) else value
                for value in kwds.values()
            ]
        )
        # Create a dicionary of arrays
        return dict(zip(kwds.keys(), map(np.array, zip(*scenarios))))

    @classmethod
    @abstractmethod
    def _calc_disp_avg(cls, **inputs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def calc_disp_avg(cls, **kwds):
        cls._check_params(**kwds)

        # index = pd.MultiIndex.from_product(kwds.values(), names=kwds.keys())
        inputs = cls._vectorize_inputs(**kwds)

        df = cls._calc_disp_avg(**inputs)

        return df
