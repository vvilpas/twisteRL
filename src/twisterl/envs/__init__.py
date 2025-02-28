# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from twisterl import twisterl_rs

Puzzle = twisterl_rs.env.Puzzle


from twisterl.utils import dynamic_import

class PyEnv:
    def __init__(self, pyenv_cls, **env_config):
        env_cls = dynamic_import(pyenv_cls)
        self.rs_env = twisterl_rs.env.PyEnv(env_cls(**env_config))
    
    def __getattr__(self, name):
        # Called when an attribute/method is not found in PyCls
        # Redirect calls to the RustCls instance
        attr = getattr(self.rs_env, name)
        if callable(attr):
            # If the attribute is callable, return it as a method
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)
            return wrapper
        else:
            # Otherwise, return it as is
            return attr

    @property
    def difficulty(self):
        return self.rs_env.difficulty

    @difficulty.setter
    def difficulty(self, new_value):
        self.rs_env.difficulty = new_value

    def __dir__(self):
        # Override dir() to include attributes and methods of RustCls
        return super().__dir__() + dir(self.rs_env)
    