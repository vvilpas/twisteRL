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
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::rl::env::Env;
use crate::nn::policy::{sample_from_logits, Policy};


pub trait PPO: Env + Clone + Sync {
    fn single_collect_ppo(&self, policy: &Policy, gamma: f32, lambda: f32) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) {
        // Init data vecs
        let mut obss: Vec<Vec<usize>> = vec![];
        let mut log_probs: Vec<Vec<f32>> = vec![];
        let mut vals: Vec<f32> = vec![];
        let mut rews: Vec<f32> = vec![];
        let mut acts: Vec<usize> = vec![];

        // Get a random initial state
        let mut state = (*self).clone();
        state.reset();

        // Loop until a final state
        loop {
            // Get current reward, masks and obs from state
            let rew = state.reward();
            let mask = state.masks();
            let obs = state.observe();

            // Calculate probs and val for current state
            let (log_prob, val) = policy.forward(obs.clone(), mask);

            // Select next action and get current value
            let act = sample_from_logits(&log_prob);

            // Store data
            obss.push(obs);
            log_probs.push(log_prob);
            vals.push(val);
            rews.push(rew);
            acts.push(act);

            // Break if we are in a final state
            if state.is_final() {
                break;
            }

            // Move to next state
            state.step(act);
        }


        // Post process rewards to get advantages and returns
        let mut advs: Vec<f32> = vec![0.0; rews.len()];
        let mut rets: Vec<f32> = vec![0.0; rews.len()];

        // Fill the final value for the advs and rets
        advs[rews.len() - 1] = rews[rews.len() - 1] - vals[rews.len() - 1];
        rets[rews.len() - 1] = rews[rews.len() - 1];
        
        // Fill the rest
        for i_t in 1..rews.len() {
            let t = rews.len() - 1 - i_t;
            
            rets[t] = rews[t] + gamma * (vals[t + 1] + lambda * advs[t + 1]);
            advs[t] = rets[t] - vals[t];
        }
        
        // Return data
        (obss, log_probs, vals, rews, acts, rets, advs)
    }

    fn collect_ppo(&self, policy: &Policy, num_episodes: usize, gamma: f32, lambda: f32, num_cores: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) {
        if num_cores == 1 {
            merge(
                (0..num_episodes).into_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect_ppo(policy, gamma, lambda)) // For each item in the range run a collection
                    .collect()
            )
        } else {
            let pool: rayon::ThreadPool = ThreadPoolBuilder::new().num_threads(num_cores).build().unwrap();
            // Use the thread pool to run the generation in parallel
            merge(pool.install(|| {
                (0..num_episodes).into_par_iter()  // Create a parallel iterator over the range 0..num_episodes
                    .map(|_| self.single_collect_ppo(policy, gamma, lambda)) // For each item in the range run a collection
                    .collect()
            }))
        }
    }

}


fn merge(mut chunks: Vec<(Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>)>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) {
    // If there's only one chunk, return it directly
    if chunks.len() == 1 {
        return chunks.pop().unwrap();
    }

    let total_size: usize = chunks.iter().map(|chunk| chunk.0.len()).sum();


    let mut merged: (Vec<Vec<usize>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) = (
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),
        Vec::with_capacity(total_size),        
    );

    for mut chunk in chunks {
        merged.0.append(&mut chunk.0);
        merged.1.append(&mut chunk.1);
        merged.2.append(&mut chunk.2);
        merged.3.append(&mut chunk.3);
        merged.4.append(&mut chunk.4);
        merged.5.append(&mut chunk.5);
        merged.6.append(&mut chunk.6);
    } 

    merged
}
