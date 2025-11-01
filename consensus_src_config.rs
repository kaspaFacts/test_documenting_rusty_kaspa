//! Configuration types and network parameters.  
//!  
//! This module re-exports all configuration types from [`kaspa_consensus_core::config`].  
//! See that module for detailed documentation on:  
//!  
//! - [`Config`] - Main configuration struct with consensus and operational parameters  
//! - [`ConfigBuilder`] - Builder pattern for constructing configurations  
//! - [`Params`] - Network-specific consensus parameters  
//! - [`ForkActivation`] - Hard fork activation management  
//! - Genesis block constants for all networks  
  
// Re-exports from consensus core for internal crate usage  
pub use kaspa_consensus_core::config::*;
