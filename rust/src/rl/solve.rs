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

use crate::rl::env::Env;
use super::search::Search;
use crate::nn::policy::{sample, argmax, Policy};

pub trait Solve: Env + Search {
    fn single_solve(&self, policy: &Policy, deterministic: bool, num_mcts_searches: usize, C: f32, max_expand_depth: usize) -> ((f32, f32), Vec<usize>) {
        let mut total_val = 0.0;
        let mut state = (*self).clone();
        let mut solution: Vec<usize> = vec![];

        while !state.is_final() {
            let val = state.reward();
            let masks = state.masks();
            let obs = state.observe();
            total_val += val;
            
            let probs = if num_mcts_searches==0 {policy.predict(obs, masks).0} else {state.predict_probs_mcts(policy, num_mcts_searches, C, max_expand_depth)};

            let action = if deterministic {argmax(&probs)} else {sample(&probs)};
            state.step(action);
            solution.push(action);
        }

        let val = state.reward();
        total_val += val;

        (((val == 1.0) as usize as f32, total_val), solution)
    }

    fn solve(&self, policy: &Policy, deterministic: bool, num_searches: usize, num_mcts_searches: usize, C: f32, max_expand_depth: usize) -> ((f32, f32), Vec<usize>) {
        let mut best: ((f32, f32), Vec<usize>) = ((0.0, -1e9), vec![]);

        for _ in 0..num_searches {
            let next_val = self.single_solve(policy, deterministic, num_mcts_searches, C, max_expand_depth);
            if next_val.0 > best.0 {
                best = next_val;
            }
        }
        best
    }
}