//! # Kaspa Consensus Implementation  
//!  
//! This crate provides the core consensus engine for the Kaspa blockchain, implementing  
//! the GHOSTDAG protocol and a multi-stage block processing pipeline.  
//!  
//! ## Overview  
//!  
//! The consensus crate is responsible for validating blocks, maintaining the DAG structure,  
//! computing the virtual state, and managing the UTXO set. It implements Kaspa's unique  
//! blockDAG consensus mechanism which allows for high block rates (1-10 BPS) while  
//! maintaining security and finality guarantees.  
//!  
//! ## Architecture  
//!  
//! The consensus system is organized into several key modules:  
//!  
//! ### Core Components  
//!  
//! - [`config`] - Configuration management including network parameters and genesis blocks  
//! - [`consensus`] - The main consensus engine coordinating all processing stages  
//! - [`params`] - Network-specific parameters (mainnet, testnet, devnet, simnet)  
//! - [`constants`] - Consensus constants and limits  
//!  
//! ### Processing Pipeline  
//!  
//! The [`pipeline`] module implements a four-stage parallel processing architecture:  
//!  
//! 1. **Header Processing** - Validates block headers and computes GHOSTDAG data  
//! 2. **Body Processing** - Validates transactions and block bodies  
//! 3. **Virtual State Processing** - Resolves the virtual state and UTXO set  
//! 4. **Pruning Processing** - Manages pruning point advancement  
//!  
//! Each stage runs in dedicated thread pools with cross-stage communication via channels.  
//!  
//! ### Consensus Processes  
//!  
//! The [`processes`] module contains the core consensus algorithms:  
//!  
//! - **GHOSTDAG** - Block ordering and blue/red classification  
//! - **Difficulty Adjustment** - Dynamic difficulty calculation (DAA)  
//! - **Reachability** - DAG reachability queries and graph traversal  
//! - **Transaction Validation** - UTXO validation and script execution  
//! - **Coinbase** - Block reward calculation with deflationary schedule  
//! - **Pruning** - Pruning point selection and proof generation  
//!  
//! ### Data Model  
//!  
//! The [`model`] module defines the storage layer and data structures:  
//!  
//! - Database stores for headers, blocks, UTXO sets, and DAG metadata  
//! - Reachability and relations services for graph queries  
//! - Virtual state management and UTXO diff tracking  
//!  
//! ### Error Handling  
//!  
//! The [`errors`] module provides comprehensive error types for:  
//!  
//! - Block validation failures (header, body, UTXO)  
//! - Transaction validation errors  
//! - Consensus rule violations  
//! - Database and storage errors  
//!  
//! ## Usage Example  
//!  
//! ```rust,no_run  
//! use kaspa_consensus::{  
//!     config::Config,  
//!     consensus::Consensus,  
//!     params::MAINNET_PARAMS,  
//! };  
//! use kaspa_database::prelude::ConnBuilder;  
//! use std::sync::Arc;  
//!  
//! // Create consensus instance  
//! let config = Config::new(MAINNET_PARAMS);  
//! let db = Arc::new(/* database setup */);  
//! let consensus = Arc::new(Consensus::new(  
//!     db,  
//!     Arc::new(config),  
//!     /* ... other parameters ... */  
//! ));  
//!  
//! // Initialize and start processing  
//! consensus.init();  
//! ```  
//!  
//! ## Key Features  
//!  
//! - **High Throughput**: Supports 1-10 blocks per second through parallel processing  
//! - **GHOSTDAG Protocol**: Provides total ordering of blocks in a DAG structure  
//! - **Fork Management**: Handles hard fork activations (e.g., Crescendo upgrade)  
//! - **Pruning Support**: Efficient storage through pruning with cryptographic proofs  
//! - **Multi-threaded**: Parallel validation across dedicated thread pools  
//!  
//! ## Related Crates  
//!  
//! - [`kaspa_consensus_core`] - Core consensus types and traits  
//! - [`kaspa_consensus_notify`] - Consensus event notification system  
//! - [`kaspa_database`] - Database abstraction layer  
//! - [`kaspa_hashes`] - Cryptographic hash functions  
//!  
//! ## Testing  
//!  
//! The [`test_helpers`] module provides utilities for consensus testing including  
//! test consensus instances and block builders.

// Until the codebase stables up, we will have a lot of these -- ignore for now
// TODO: remove this
#![allow(dead_code)]

pub mod config;
pub mod consensus;
pub mod constants;
pub mod errors;
pub mod model;
pub mod params;
pub mod pipeline;
pub mod processes;
pub mod test_helpers;
