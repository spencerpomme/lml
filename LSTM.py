#!/anaconda/envs/tensorflow/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:55 2018

@author: ZHANG PINGCHENG
"""


def flatten(nested): return list(filter(lambda _: _, (lambda _: ((yield from flatten(
    e)) if isinstance(e, Iterable) else (yield round(e, 6)) for e in _))(nested)))
