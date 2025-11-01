//! # Consensus Configuration System  
//!  
//! This module provides configuration types and network parameters for the Kaspa consensus engine.  
//! It includes the main [`Config`] struct, network-specific parameters, fork activation management,  
//! and genesis block definitions for all supported networks.  
//!  
//! ## Overview  
//!  
//! The configuration system is designed to support multiple networks (mainnet, testnet, devnet, simnet)  
//! with different consensus parameters, while also handling hard fork activations like the Crescendo  
//! upgrade which transitions the network from 1 BPS to 10 BPS.  
//!  
//! ## Core Types  
//!  
//! ### Configuration Management  
//!  
//! - [`Config`] - Main configuration struct bundling consensus parameters, performance settings,  
//!   and operational flags  
//! - [`ConfigBuilder`] - Builder pattern for constructing `Config` instances with custom settings  
//! - [`Params`] - Network-specific consensus parameters (GHOSTDAG K, difficulty windows, etc.)  
//! - [`PerfParams`] - Performance tuning parameters for cache sizes and thread pools  
//!  
//! ### Fork Activation System  
//!  
//! - [`ForkActivation`] - Manages hard fork activation at specific DAA scores  
//! - [`ForkedParam<T>`] - Wraps parameters that change values at fork boundaries  
//! - [`CrescendoParams`] - Parameters for the Crescendo hard fork (10 BPS upgrade)  
//!  
//! The fork activation system allows consensus parameters to change at specific DAA scores,  
//! enabling network upgrades without requiring all nodes to upgrade simultaneously.  
//!  
//! ### Network Parameters  
//!  
//! Pre-configured parameter sets for each network:  
//!  
//! - [`MAINNET_PARAMS`] - Production network (1 BPS, Crescendo activation at DAA 110,165,000)  
//! - [`TESTNET_PARAMS`] - Test network (1 BPS, Crescendo activation at DAA 88,657,000)  
//! - [`SIMNET_PARAMS`] - Simulation network (10 BPS, Crescendo always active)  
//! - [`DEVNET_PARAMS`] - Development network (1 BPS, Crescendo not activated)  
//!  
//! ### Genesis Blocks  
//!  
//! - [`GenesisBlock`] - Genesis block structure with hash, merkle root, and coinbase payload  
//! - [`GENESIS`] - Mainnet genesis block  
//! - [`TESTNET_GENESIS`] - Testnet genesis block  
//! - [`TESTNET11_GENESIS`] - Testnet 11 genesis block (10 BPS)  
//! - [`SIMNET_GENESIS`] - Simnet genesis block  
//! - [`DEVNET_GENESIS`] - Devnet genesis block  
//!  
//! ### BPS Calculations  
//!  
//! - [`Bps<const BPS: u64>`] - Generic BPS calculator for deriving network parameters  
//! - [`TenBps`] - Type alias for 10 BPS networks (used by Crescendo)  
//! - [`calculate_ghostdag_k()`] - Calculates GHOSTDAG K parameter from network delay and BPS  
//!  
//! ## Usage Examples  
//!  
//! ### Basic Configuration  
//!  
//! ```rust  
//! use kaspa_consensus_core::config::{Config, MAINNET_PARAMS};  
//!  
//! // Create a basic mainnet configuration  
//! let config = Config::new(MAINNET_PARAMS);  
//! ```  
//!  
//! ### Using ConfigBuilder  
//!  
//! ```rust  
//! use kaspa_consensus_core::config::{ConfigBuilder, MAINNET_PARAMS};  
//!  
//! // Build a custom configuration  
//! let config = ConfigBuilder::new(MAINNET_PARAMS)  
//!     .adjust_perf_params_to_consensus_params()  
//!     .skip_proof_of_work()  // For testing  
//!     .set_archival()        // Enable archival mode  
//!     .build();  
//! ```  
//!  
//! ### Editing Consensus Parameters  
//!  
//! ```rust  
//! use kaspa_consensus_core::config::{ConfigBuilder, ForkActivation, DEVNET_PARAMS};  
//!  
//! // Customize consensus parameters for testing  
//! let config = ConfigBuilder::new(DEVNET_PARAMS)  
//!     .edit_consensus_params(|params| {  
//!         params.crescendo_activation = ForkActivation::always();  
//!         params.prior_coinbase_maturity = 10;  
//!     })  
//!     .build();  
//! ```  
//!  
//! ### Applying Runtime Arguments  
//!  
//! ```rust  
//! use kaspa_consensus_core::config::{Config, ConfigBuilder, MAINNET_PARAMS};  
//!  
//! let config = ConfigBuilder::new(MAINNET_PARAMS)  
//!     .apply_args(|config| {  
//!         config.utxoindex = true;  
//!         config.is_archival = true;  
//!         config.ram_scale = 2.0;  
//!     })  
//!     .build();  
//! ```  
//!  
//! ## Fork Activation  
//!  
//! The Crescendo hard fork transitions the network from 1 BPS to 10 BPS, changing multiple  
//! consensus parameters:  
//!  
//! - **GHOSTDAG K**: Increases to accommodate higher block rate  
//! - **Difficulty Window**: Switches from full window to sampled window  
//! - **Past Median Time**: Switches to sampled calculation  
//! - **Transaction Limits**: Reduces max inputs/outputs for mass calculation efficiency  
//!  
//! Parameters that change at fork activation are wrapped in [`ForkedParam<T>`], which provides  
//! methods to query the active value at any DAA score:  
//!  
//! ```rust  
//! use kaspa_consensus_core::config::MAINNET_PARAMS;  
//!  
//! let params = MAINNET_PARAMS;  
//! let current_daa_score = 110_000_000;  
//!  
//! // Get the active GHOSTDAG K value  
//! let k = params.ghostdag_k().get(current_daa_score);  
//!  
//! // Check if fork is active  
//! if params.crescendo_activation.is_active(current_daa_score) {  
//!     // Use post-fork parameters  
//! }  
//! ```  
//!  
//! ## Configuration Fields  
//!  
//! ### Consensus Parameters  
//!  
//! The [`Config`] struct includes consensus-sensitive parameters from [`Params`]:  
//!  
//! - **Network Identity**: `net`, `genesis`, `dns_seeders`  
//! - **GHOSTDAG**: `ghostdag_k()`, `max_block_parents()`, `mergeset_size_limit()`  
//! - **Difficulty Adjustment**: `difficulty_window_size()`, `difficulty_sample_rate()`  
//! - **Timing**: `target_time_per_block()`, `timestamp_deviation_tolerance`  
//! - **Finality**: `finality_depth()`, `pruning_depth()`, `merge_depth()`  
//! - **Transactions**: `max_tx_inputs()`, `max_tx_outputs()`, `max_block_mass`  
//! - **Coinbase**: `coinbase_maturity()`, `deflationary_phase_daa_score`  
//!  
//! ### Operational Flags  
//!  
//! Non-consensus configuration options:  
//!  
//! - `process_genesis` - Whether to process the genesis block on initialization  
//! - `is_archival` - Archival nodes keep all historical data  
//! - `enable_sanity_checks` - Enable compute-intensive validation checks  
//! - `utxoindex` - Maintain UTXO index for efficient queries  
//! - `unsafe_rpc` - Enable RPC commands that affect node state  
//! - `enable_unsynced_mining` - Allow mining while not synced (for network initialization)  
//! - `enable_mainnet_mining` - Explicit flag required for mainnet mining  
//! - `ram_scale` - Scale factor for memory allocation bounds  
//! - `retention_period_days` - Data retention period for non-archival nodes  
//!  
//! ### Performance Parameters  
//!  
//! The [`PerfParams`] struct controls performance-critical settings:  
//!  
//! - `header_data_cache_size` - Cache size for header-related data  
//! - `block_data_cache_size` - Cache size for block body data  
//! - `utxo_set_cache_size` - Cache size for UTXO data  
//! - `block_window_cache_size` - Cache size for difficulty/median time windows  
//! - `block_processors_num_threads` - Thread pool size for block processors  
//! - `virtual_processor_num_threads` - Thread pool size for virtual processor  
//!  
//! ## Submodules  
//!
//! - [`bps`] - BPS-dependent parameter calculations
//! - [`constants`] - Consensus constants and limits
//! - [`genesis`] - Genesis block definitions
//! - [`params`] - Network parameter structures and constants
//!
//! ## Notes
//!
//! The `Config` struct implements `Deref<Target = Params>`, allowing direct access to consensus
//! parameters without explicit field access. This means you can call `config.ghostdag_k()` instead
//! of `config.params.ghostdag_k()`.
  
pub mod bps;
pub mod constants;
pub mod genesis;
pub mod params;

use kaspa_utils::networking::{ContextualNetAddress, NetAddress};

#[cfg(feature = "devnet-prealloc")]
use crate::utxo::utxo_collection::UtxoCollection;
#[cfg(feature = "devnet-prealloc")]
use std::sync::Arc;

use std::ops::Deref;

use {
    constants::perf::{PerfParams, PERF_PARAMS},
    params::Params,
};

/// Various consensus configurations all bundled up under a single struct. Use `Config::new` for directly building from
/// a `Params` instance. For anything more complex it is recommended to use `ConfigBuilder`. NOTE: this struct can be
/// implicitly de-refed into `Params`
#[derive(Clone, Debug)]
pub struct Config {
    /// Consensus params
    pub params: Params,
    /// Performance params
    pub perf: PerfParams,

    //
    // Additional consensus configuration arguments which are not consensus sensitive
    //
    pub process_genesis: bool,

    /// Indicates whether this node is an archival node
    pub is_archival: bool,

    /// Enable various sanity checks which might be compute-intensive (mostly performed during pruning)
    pub enable_sanity_checks: bool,

    // TODO: move non-consensus parameters like utxoindex to a higher scoped Config
    /// Enable the UTXO index
    pub utxoindex: bool,

    /// Enable RPC commands which affect the state of the node
    pub unsafe_rpc: bool,

    /// Allow the node to accept blocks from RPC while not synced
    /// (required when initiating a new network from genesis)
    pub enable_unsynced_mining: bool,

    /// Allow mainnet mining. Until a stable Beta version we keep this option off by default
    pub enable_mainnet_mining: bool,

    pub user_agent_comments: Vec<String>,

    /// If undefined, sets it to 0.0.0.0
    pub p2p_listen_address: ContextualNetAddress,

    pub externalip: Option<NetAddress>,

    pub block_template_cache_lifetime: Option<u64>,

    #[cfg(feature = "devnet-prealloc")]
    pub initial_utxo_set: Arc<UtxoCollection>,

    pub disable_upnp: bool,

    /// A scale factor to apply to memory allocation bounds
    pub ram_scale: f64,

    /// The number of days to keep data for
    pub retention_period_days: Option<f64>,
}

impl Config {
    pub fn new(params: Params) -> Self {
        Self::with_perf(params, PERF_PARAMS)
    }

    pub fn with_perf(params: Params, perf: PerfParams) -> Self {
        Self {
            params,
            perf,
            process_genesis: true,
            is_archival: false,
            enable_sanity_checks: false,
            utxoindex: false,
            unsafe_rpc: false,
            enable_unsynced_mining: false,
            enable_mainnet_mining: false,
            user_agent_comments: Default::default(),
            externalip: None,
            p2p_listen_address: ContextualNetAddress::unspecified(),
            block_template_cache_lifetime: None,

            #[cfg(feature = "devnet-prealloc")]
            initial_utxo_set: Default::default(),
            disable_upnp: false,
            ram_scale: 1.0,
            retention_period_days: None,
        }
    }

    pub fn to_builder(&self) -> ConfigBuilder {
        ConfigBuilder { config: self.clone() }
    }
}

impl AsRef<Params> for Config {
    fn as_ref(&self) -> &Params {
        &self.params
    }
}

impl Deref for Config {
    type Target = Params;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    pub fn new(params: Params) -> Self {
        Self { config: Config::new(params) }
    }

    pub fn set_perf_params(mut self, perf: PerfParams) -> Self {
        self.config.perf = perf;
        self
    }

    pub fn adjust_perf_params_to_consensus_params(mut self) -> Self {
        self.config.perf.adjust_to_consensus_params(&self.config.params);
        self
    }

    pub fn edit_consensus_params<F>(mut self, edit_func: F) -> Self
    where
        F: Fn(&mut Params),
    {
        edit_func(&mut self.config.params);
        self
    }

    pub fn apply_args<F>(mut self, edit_func: F) -> Self
    where
        F: Fn(&mut Config),
    {
        edit_func(&mut self.config);
        self
    }

    pub fn skip_proof_of_work(mut self) -> Self {
        self.config.params.skip_proof_of_work = true;
        self
    }

    pub fn set_archival(mut self) -> Self {
        self.config.is_archival = true;
        self
    }

    pub fn enable_sanity_checks(mut self) -> Self {
        self.config.enable_sanity_checks = true;
        self
    }

    pub fn skip_adding_genesis(mut self) -> Self {
        self.config.process_genesis = false;
        self
    }

    pub fn build(self) -> Config {
        self.config
    }
}
