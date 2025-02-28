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

macro_rules! impl_algorithms {
    ($struct_name:ident) => {
        use crate::rl::search::Search;
        use crate::rl::evaluate::Evaluate;
        use crate::rl::solve::Solve;
        use crate::rl::ppo::PPO;
        use crate::rl::az::AZ;
        use crate::nn::policy::Policy;

        impl Search for $struct_name{}
        impl Evaluate for $struct_name{}
        impl Solve for $struct_name{}
        impl PPO for $struct_name{}
        impl AZ for $struct_name{}

        #[pymethods]
        impl $struct_name {
            #[getter]
            fn get_difficulty(&self) -> PyResult<usize> {
                Ok(self.difficulty)
            }

            #[setter]
            fn set_difficulty(&mut self, value: usize) -> PyResult<()> {
                self.difficulty = value;
                Ok(())
            }

            // From Env
            pub fn num_actions(&self) -> usize {
                <Self as Env>::num_actions(self)
            }

            pub fn obs_shape(&self) -> Vec<usize> {
                <Self as Env>::obs_shape(self)
            }

            pub fn set_state(&mut self, state: Vec<i64>) {
                <Self as Env>::set_state(self, state);
            }

            pub fn reset(&mut self) {
                <Self as Env>::reset(self);
            }

            pub fn step(&mut self, action: usize) {
                <Self as Env>::step(self, action);
            }

            pub fn masks(&self) -> Vec<bool> {
                <Self as Env>::masks(self)
            }

            pub fn is_final(&self) -> bool {
                <Self as Env>::is_final(self)
            }

            pub fn reward(&self) -> f32 {
                <Self as Env>::reward(self)
            }

            pub fn observe(&self) -> Vec<usize> {
                <Self as Env>::observe(self)
            }

            pub fn twists(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
                <Self as Env>::twists(self)
            }
            

            // From MCTS
            pub fn predict_probs_mcts(&self, policy: &Policy, num_mcts_searches: usize, C: f32, max_expand_depth: usize) -> Vec<f32> {
                <Self as Search>::predict_probs_mcts(&self, policy, num_mcts_searches, C, max_expand_depth)
            }

            // From Solve
            pub fn solve(&self, policy: &Policy, deterministic: bool, num_searches: usize, num_mcts_searches: usize, C: f32, max_expand_depth: usize) -> ((f32, f32), Vec<usize>) {
                <Self as Solve>::solve(self, policy, deterministic, num_searches, num_mcts_searches, C, max_expand_depth)
            }

            // From Evaluate
            pub fn evaluate(&self, policy: &Policy, num_episodes: usize, deterministic: bool, num_searches: usize, num_mcts_searches: usize, seed: usize, C: f32, max_expand_depth: usize, num_cores: usize) -> (f32, f32) {
                <Self as Evaluate>::evaluate(self, policy, num_episodes, deterministic, num_searches, num_mcts_searches, seed, C, max_expand_depth, num_cores)
            }

            // From PPO
            pub fn collect_ppo(&self, policy: &Policy, num_episodes: usize, gamma: f32, lambda: f32, num_cores: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) {
                <Self as PPO>::collect_ppo(self, policy, num_episodes, gamma, lambda, num_cores)
            }

            // From AZ
            pub fn collect_az(&self, policy: &Policy, num_episodes: usize, num_mcts_searches: usize, C: f32, max_expand_depth: usize, num_cores: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>) {
                <Self as AZ>::collect_az(&self, policy, num_episodes, num_mcts_searches, C, max_expand_depth, num_cores)
            }
        }
    };
}