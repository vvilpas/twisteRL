// -*- coding: utf-8 -*-
/* 
(C) Copyright 2025 IBM. All Rights Reserved.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
*/

use std::borrow::Borrow;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use crate::rl::env::Env;



#[pyclass]
pub struct PyEnv {
    py_env: Py<PyAny>, // Reference to the Python object implementing the environment
    difficulty: usize
}


impl Clone for PyEnv {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            PyEnv {
                py_env: self.py_env.call_method0(py, "copy").expect("Python `copy` method failed."),
                difficulty: self.difficulty
            }
        })
    }
}


#[pymethods]
impl PyEnv {
    /// Constructor: Create a new `PythonEnv` from a Python object
    #[new]
    pub fn new(py_env: Py<PyAny>) -> Self {
        PyEnv { py_env, difficulty:1 }
    }
}

impl Env for PyEnv {
    // Sets the current difficulty
    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    // Returns current difficulty
    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env.call_method1(py, "set_state", (state,))
                .expect("Python `set_state` method failed.");
        });
    }

    fn num_actions(&self) -> usize {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "num_actions")
                .and_then(|val| val.extract::<usize>(py))
                .expect("Python `num_actions` method must return an integer.")
        })
    }

    fn obs_shape(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "obs_shape")
                .and_then(|val| val.extract::<Vec<usize>>(py))
                .expect("Python `obs_shape` method must return a list of integers.")
        })
    }
    
    fn reset(&mut self) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method1(py, "reset", (self.difficulty,))
                .expect("Python `reset` method failed.");
        });
    }

    fn step(&mut self, action: usize) {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method1(py, "next", (action,))
                .expect("Python `next` method failed.");
        });
    }

    fn masks(&self) -> Vec<bool> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "masks")
                .and_then(|val| val.extract::<Vec<bool>>(py))
                .expect("Python `masks` method must return a list of booleans.")
        })
    }

    fn is_final(&self) -> bool {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "is_final")
                .and_then(|val| val.extract::<bool>(py))
                .expect("Python `is_final` method must return a boolean.")
        })
    }

    fn reward(&self) -> f32 {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "value")
                .and_then(|val| val.extract::<f32>(py))
                .expect("Python `value` method must return a float.")
        })
    }

    fn observe(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let py_env = self.py_env.borrow();
            py_env
                .call_method0(py, "observe")
                .and_then(|val| val.extract::<Vec<usize>>(py))
                .expect("Python `observe` method must return a list of integers.")
        })
    }
}


// Dont forget this line! this implements all the RL algorithms for your env
impl_algorithms!(PyEnv);