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

use crate::rl::tree::Tree;
use crate::nn::policy::{sample, Policy};


// MCTS node data 
pub struct MCTSNode<T: Env> {
    state: T,
    action_taken: Option<usize>,
    prior: f32,
    pub visit_count: u32,
    pub value_sum: f32,
}
impl<T: Env> MCTSNode<T> {
    fn ucb(&self, child: &MCTSNode<T>, C: f32) -> f32 {
        let q_value = if child.visit_count == 0 {
            //0.5
            0.0
        } else {
            //((child.value_sum / (child.visit_count as f32)) + 1.0) / 2.0
            child.value_sum / (child.visit_count as f32)
        };

        q_value + C * ((self.visit_count as f32).sqrt() / (child.visit_count as f32 + 1.0)) * child.prior
    }
}

// MCTS specialized Tree 
type MCTSTree<T> = Tree<MCTSNode<T>>;
impl<T: Env+Clone> MCTSTree<T> {
    pub fn backpropagate(&mut self, node_idx: usize, value: f32) {
        self.nodes[node_idx].val.value_sum += value;
        self.nodes[node_idx].val.visit_count += 1;

        if let Some(parent_idx) = self.nodes[node_idx].parent {
            self.backpropagate(parent_idx, value);
        }
    }

    // Expand the node by adding child nodes based on the policy probabilities
    pub fn expand(&mut self, node_idx: usize, action_probs: Vec<f32>) {
        let base_state = self.nodes[node_idx].val.state.clone();
        for (action, &prob) in action_probs.iter().enumerate() {
            if prob > 0.0 {
                let mut next_state = base_state.clone();
                next_state.step(action);

                self.add_child_to_node(
                MCTSNode{
                        state: next_state,
                        action_taken: Some(action),
                        prior: prob,
                        visit_count: 0,
                        value_sum: 0.0
                    }, 
                    node_idx
                );
            }
        }
    }

    // Select the best child node based on UCB score
    pub fn next(&self, node_idx: usize, C: f32) -> usize {
        let mut best_child = None;
        let mut best_ucb = f32::NEG_INFINITY;

        for child_idx in self.nodes[node_idx].children.iter() {
            let ucb = self.nodes[node_idx].val.ucb(&self.nodes[*child_idx].val, C);

            if ucb > best_ucb {
                best_child = Some(*child_idx);
                best_ucb = ucb;
            }
        }

        best_child.expect("No child selected in next()")
    }

    // Select the best child node based on UCB score
    pub fn next_sample(&self, node_idx: usize) -> usize {
        let probs: Vec<f32> = self.nodes[node_idx].children.iter().map(
            |&child_idx| self.nodes[child_idx].val.prior
        ).collect();
        let child_num = sample(&probs);
        self.nodes[node_idx].children[child_num]
    }
}


pub trait Search: Env + Clone {
    // Perform the MCTS search starting from the given state
    fn predict_probs_mcts(&self, policy: &Policy, num_mcts_searches: usize, C: f32, max_expand_depth: usize) -> Vec<f32> {
        // Get the initial policy and value from the neural network
        let (action_probs, _) = policy.full_predict(self.observe(), self.masks());

        // Create the tree and root node
        let mut tree: Tree<MCTSNode<Self>> = Tree::new();
        let root_idx = tree.new_node(
            MCTSNode{
                state: self.clone(),
                action_taken: None,
                prior: 0.0,
                visit_count: 1,
                value_sum: 0.0
            }
        );

        // Expand the root node
        tree.expand(root_idx, action_probs);

        // Perform the search iterations
        for _ in 0..num_mcts_searches {
            let mut node_idx = root_idx.clone();

            // First expand until leaf node
            while tree.nodes[node_idx].children.len() > 0 {
                node_idx = tree.next(node_idx, C);
            }
            let mut value: f32 = 0.0;
            let mut expanded_depth = 0;  // This counts how many steps we expand at most

            // Then add nodes until an end state
            while expanded_depth < max_expand_depth {
                // Get value
                let node_state = &tree.nodes[node_idx].val.state;
                value = node_state.reward();

                // Break if is_final
                if node_state.is_final() {
                    break;
                }

                // If not, predict actions, expand tree and select by sampling
                let (action_probs, new_value) =
                    policy.full_predict(node_state.observe(), node_state.masks());
                tree.expand(node_idx, action_probs);
                node_idx = tree.next_sample(node_idx);
                value = new_value;
                expanded_depth += 1;
            }

            // Backpropagation
            tree.backpropagate(node_idx, value);
        }

        // Calculate the action probabilities from the root's children
        let mut mcts_action_probs = vec![0.0; self.num_actions() as usize];

        for child_idx in tree.nodes[root_idx].children.iter() {
            let action_taken = tree.nodes[*child_idx].val.action_taken.unwrap() as usize;
            mcts_action_probs[action_taken] = tree.nodes[*child_idx].val.visit_count as f32;  // This is classical mcts
        }

        let sum_probs: f32 = mcts_action_probs.iter().sum();
        if sum_probs > 0.0 {
            for p in mcts_action_probs.iter_mut() {
                *p /= sum_probs;
            }
        } else {
            mcts_action_probs = vec![1.0/(self.num_actions() as f32); self.num_actions() as usize];
        }

        mcts_action_probs
    }
}

