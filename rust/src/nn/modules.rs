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

use nalgebra::DVector;

use crate::nn::layers::Linear;

#[derive(Clone)]
pub struct Sequential {
    layers: Vec<Box<Linear>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<Linear>>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, input: DVector<f32>) -> DVector<f32> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }
}
