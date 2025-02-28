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

use std::usize;

use crate::rl::{env::Env, search::Search};
use crate::nn::policy::{sample, Policy};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;


pub trait AZ: Env + Search + Sync {
    fn single_collect_az(&self, policy: &Policy, num_mcts_searches: usize, C: f32, max_expand_depth: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>) {
        // Init data vecs
        let mut obs: Vec<Vec<usize>> = vec![];
        let mut probs: Vec<Vec<f32>> = vec![];
        let mut vals: Vec<f32> = vec![];
        let mut total_vals: Vec<f32> = vec![];


        // Get a random initial state
        let mut state = (*self).clone();
        state.reset();

        let mut total_val = 0.0;

        // Loop until a final state
        loop {
            // Calculate MCTS probs for current state
            let mcts_probs = state.predict_probs_mcts(policy, num_mcts_searches, C, max_expand_depth);

            // Select next action and get current value
            let action = sample(&mcts_probs);
            let val = state.reward();
            total_vals.push(total_val);

            total_val += val;

            // Store data
            obs.push(state.observe());
            probs.push(mcts_probs);
            vals.push(val);

            // Break if we are in a final state
            if state.is_final() {
                break;
            }

            // Move to next state
            state.step(action);
        }


        // Post process rewards
        let remaining_vals: Vec<f32> = total_vals.iter().map(|&v| total_val - v).collect();
        
        // Return data
        (obs, probs, remaining_vals)
    }

     fn collect_az(&self, policy: &Policy, num_episodes: usize, num_mcts_searches: usize, C: f32, max_expand_depth: usize, num_cores: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>) {
        if num_cores == 1 {
            merge(
                (0..num_episodes).into_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect_az(policy, num_mcts_searches, C, max_expand_depth)) // For each item in the range run a collection
                    .collect()
            )
        } else {
            let pool = ThreadPoolBuilder::new().num_threads(num_cores).build().unwrap();
            // Use the thread pool to run the generation in parallel
            merge(pool.install(|| {
                (0..num_episodes).into_par_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect_az(policy, num_mcts_searches, C, max_expand_depth)) // For each item in the range run a collection
                    .collect()
            }))
        }
    }
}


fn merge(mut chunks: Vec<(Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>)>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>) {
    // If there's only one chunk, return it directly
    if chunks.len() == 1 {
        return chunks.pop().unwrap();
    }

    let total_size: usize = chunks.iter().map(|chunk| chunk.0.len()).sum();


    let mut merged: (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>) = (
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),        
    );

    for mut chunk in chunks {
        merged.0.append(&mut chunk.0);
        merged.1.append(&mut chunk.1);
        merged.2.append(&mut chunk.2);
    } 

    merged
}