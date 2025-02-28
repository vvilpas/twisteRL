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

use pyo3::prelude::*;

#[macro_use]
mod macros;

pub mod nn;
pub mod rl;
pub mod envs;

// NN Module
use crate::nn::modules::Sequential;
use crate::nn::layers::{EmbeddingBag, Linear};
use crate::nn::policy::Policy;

fn init_nn_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmbeddingBag>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<Linear>()?;
    m.add_class::<Policy>()?;
    Ok(())
}

// Env Module
use crate::envs::puzzle::Puzzle;
use crate::envs::pyenv::PyEnv;

fn init_env_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Puzzle>()?;
    m.add_class::<PyEnv>()?;
    Ok(())
}


// Full package
#[pymodule]
fn twisterl_rs(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_submod = PyModule::new_bound(py, "nn")?;
    init_nn_module(&nn_submod)?;
    m.add_submodule(&nn_submod)?;

    let env_submod = PyModule::new_bound(py, "env")?;
    init_env_module(&env_submod)?;
    m.add_submodule(&env_submod)?;

    Ok(())
}
