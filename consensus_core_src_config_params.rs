pub use super::{
    bps::{Bps, TenBps},
    constants::consensus::*,
    genesis::{GenesisBlock, DEVNET_GENESIS, GENESIS, SIMNET_GENESIS, TESTNET11_GENESIS, TESTNET_GENESIS},
};
use crate::{
    constants::STORAGE_MASS_PARAMETER,
    network::{NetworkId, NetworkType},
    BlockLevel, KType,
};
use kaspa_addresses::Prefix;
use kaspa_math::Uint256;
use std::cmp::min;

/// Manages hard fork activation at specific DAA (Difficulty Adjustment Algorithm) scores.  
///  
/// A fork activation represents a point in the blockchain's history (identified by DAA score)  
/// where consensus rules change. This allows the network to upgrade without requiring all  
/// nodes to upgrade simultaneously.  
///  
/// # Special Values  
///  
/// - [`ForkActivation::never()`] - Fork is not scheduled (DAA score = u64::MAX)  
/// - [`ForkActivation::always()`] - Fork was active from genesis (DAA score = 0)  
///  
/// # Examples  
///  
/// ```  
/// use kaspa_consensus_core::config::params::ForkActivation;  
///  
/// // Crescendo fork activates at DAA score 110,165,000 on mainnet  
/// let crescendo = ForkActivation::new(110_165_000);  
///  
/// // Check if fork is active at current DAA score  
/// let current_daa = 110_200_000;  
/// assert!(crescendo.is_active(current_daa));  
///  
/// // Check if fork activated recently (within 1000 blocks)  
/// assert!(crescendo.is_within_range_from_activation(current_daa, 1000));  
/// ```  

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ForkActivation(u64);

impl ForkActivation {  
    const NEVER: u64 = u64::MAX;  
    const ALWAYS: u64 = 0;  
  
    /// Creates a new fork activation at the specified DAA score.  
    ///  
    /// # Arguments  
    ///  
    /// * `daa_score` - The DAA score at which the fork becomes active  
    pub const fn new(daa_score: u64) -> Self {  
        Self(daa_score)  
    }  
  
    /// Returns a fork activation that never occurs (set to u64::MAX).  
    ///  
    /// Used for forks that are not scheduled or have been cancelled.  
    pub const fn never() -> Self {  
        Self(Self::NEVER)  
    }  
  
    /// Returns a fork activation that was always active (set to 0).  
    ///  
    /// Used for forks that were active from the genesis block.  
    pub const fn always() -> Self {  
        Self(Self::ALWAYS)  
    }  
  
    /// Returns the actual DAA score triggering the activation.  
    ///  
    /// # Warning  
    ///  
    /// This should only be used when the explicit value is required for computations  
    /// (e.g., coinbase subsidy calculations). For activation checks, always use  
    /// [`is_active()`](Self::is_active) instead.  
    pub fn daa_score(self) -> u64 {  
        self.0  
    }  
  
    /// Checks if the fork is active at the given DAA score.  
    ///  
    /// Returns `true` if `current_daa_score >= activation_daa_score`.  
    ///  
    /// # Arguments  
    ///  
    /// * `current_daa_score` - The current DAA score to check against  
    pub fn is_active(self, current_daa_score: u64) -> bool {  
        current_daa_score >= self.0  
    }  
  
    /// Checks if the fork was "recently" activated within the provided range.  
    ///  
    /// Returns `false` for forks that were always active, since they were never activated.  
    ///  
    /// # Arguments  
    ///  
    /// * `current_daa_score` - The current DAA score  
    /// * `range` - The number of blocks to look back from current  
    ///  
    /// # Returns  
    ///  
    /// `true` if the fork activated within `range` blocks of `current_daa_score`  
    pub fn is_within_range_from_activation(self, current_daa_score: u64, range: u64) -> bool {  
        self != Self::always() && self.is_active(current_daa_score) && current_daa_score < self.0 + range  
    }  
  
    /// Checks if the fork is expected to activate "soon" within the provided range.  
    ///  
    /// # Arguments  
    ///  
    /// * `current_daa_score` - The current DAA score  
    /// * `range` - The number of blocks to look ahead from current  
    ///  
    /// # Returns  
    ///  
    /// `Some(distance)` with the number of blocks until activation if within range,  
    /// `None` otherwise  
    pub fn is_within_range_before_activation(self, current_daa_score: u64, range: u64) -> Option<u64> {  
        if !self.is_active(current_daa_score) && current_daa_score + range > self.0 {  
            Some(self.0 - current_daa_score)  
        } else {  
            None  
        }  
    }  
}

/// A consensus parameter that has different values before and after fork activation.  
///  
/// This wrapper allows parameters to change at specific DAA scores during hard forks,  
/// enabling smooth network upgrades. The parameter automatically returns the correct  
/// value based on the current DAA score.  
///  
/// # Type Parameters  
///  
/// * `T` - The parameter type (must implement `Copy`)  
///  
/// # Examples  
///  
/// ```  
/// use kaspa_consensus_core::config::params::{ForkActivation, ForkedParam};  
///  
/// // GHOSTDAG K changes from 18 to 124 at Crescendo activation  
/// let crescendo = ForkActivation::new(110_165_000);  
/// let ghostdag_k = ForkedParam::new(18, 124, crescendo);  
///  
/// // Before activation  
/// assert_eq!(ghostdag_k.get(100_000_000), 18);  
///  
/// // After activation  
/// assert_eq!(ghostdag_k.get(120_000_000), 124);  
///  
/// // Get bounds for initialization  
/// assert_eq!(ghostdag_k.lower_bound(), 18);  
/// assert_eq!(ghostdag_k.upper_bound(), 124);  
/// ```  
#[derive(Clone, Copy, Debug)]
pub struct ForkedParam<T: Copy> {
    pre: T,
    post: T,
    activation: ForkActivation,
}

impl<T: Copy> ForkedParam<T> {  
    /// Creates a new forked parameter with pre and post-activation values.  
    const fn new(pre: T, post: T, activation: ForkActivation) -> Self {  
        Self { pre, post, activation }  
    }  
  
    /// Creates a constant parameter that never changes (activation set to never).  
    ///  
    /// Useful for parameters that don't change during forks but need to be  
    /// wrapped in `ForkedParam` for API consistency.  
    pub const fn new_const(val: T) -> Self {  
        Self { pre: val, post: val, activation: ForkActivation::never() }  
    }  
  
    /// Returns the fork activation point for this parameter.  
    pub fn activation(&self) -> ForkActivation {  
        self.activation  
    }  
  
    /// Gets the active value at the specified DAA score.  
    ///  
    /// Returns `post` if the fork is active, `pre` otherwise.  
    ///  
    /// # Arguments  
    ///  
    /// * `daa_score` - The DAA score to query  
    pub fn get(&self, daa_score: u64) -> T {  
        if self.activation.is_active(daa_score) {  
            self.post  
        } else {  
            self.pre  
        }  
    }  
  
    /// Returns the value before activation.  
    ///  
    /// If activation is set to `always()`, returns `post` since the fork  
    /// was active from genesis.  
    pub fn before(&self) -> T {  
        match self.activation.0 {  
            ForkActivation::ALWAYS => self.post,  
            _ => self.pre,  
        }  
    }  
  
    /// Returns the permanent long-term value after activation.  
    ///  
    /// If activation is set to `never()`, returns `pre` since the fork  
    /// will never activate.  
    pub fn after(&self) -> T {  
        match self.activation.0 {  
            ForkActivation::NEVER => self.pre,  
            _ => self.post,  
        }  
    }  
  
    /// Maps this `ForkedParam<T>` to a new `ForkedParam<U>` by applying a function.  
    ///  
    /// The function is applied to both pre and post values, preserving the  
    /// activation point.  
    ///  
    /// # Arguments  
    ///  
    /// * `f` - Function to apply to both values  
    pub fn map<U: Copy, F: Fn(T) -> U>(&self, f: F) -> ForkedParam<U> {  
        ForkedParam::new(f(self.pre), f(self.post), self.activation)  
    }  
}

impl<T: Copy + Ord> ForkedParam<T> {  
    /// Returns the minimum of `pre` and `post` values.  
    ///  
    /// Useful for non-consensus initializations that need to know value bounds.  
    ///  
    /// # Special Cases  
    ///  
    /// - If activation is `never()`, returns `pre`  
    /// - If activation is `always()`, returns `post`  
    pub fn lower_bound(&self) -> T {  
        match self.activation.0 {  
            ForkActivation::NEVER => self.pre,  
            ForkActivation::ALWAYS => self.post,  
            _ => self.pre.min(self.post),  
        }  
    }  
  
    /// Returns the maximum of `pre` and `post` values.  
    ///  
    /// Useful for non-consensus initializations that need to know value bounds.  
    ///  
    /// # Special Cases  
    ///  
    /// - If activation is `never()`, returns `pre`  
    /// - If activation is `always()`, returns `post`  
    pub fn upper_bound(&self) -> T {  
        match self.activation.0 {  
            ForkActivation::NEVER => self.pre,  
            ForkActivation::ALWAYS => self.post,  
            _ => self.pre.max(self.post),  
        }  
    }  
}

/// Parameters for the Crescendo hard fork.  
///  
/// The Crescendo fork transitions the network from 1 BPS (blocks per second) to 10 BPS,  
/// requiring changes to multiple consensus parameters including GHOSTDAG K, difficulty  
/// windows, and transaction limits.  
///  
/// # Key Changes  
///  
/// - **Block Rate**: 1 BPS → 10 BPS (1000ms → 100ms per block)  
/// - **GHOSTDAG K**: 18 → 124 (to handle higher anticone sizes)  
/// - **Difficulty Window**: Full window → Sampled window (for efficiency)  
/// - **Transaction Limits**: Reduced to limit mass calculation costs  
///  
/// # Activation  
///  
/// - **Mainnet**: DAA score 110,165,000 (~May 5, 2025)  
/// - **Testnet**: DAA score 88,657,000 (~March 6, 2025)  
/// - **Simnet**: Always active (from genesis)  
/// - **Devnet**: Never active (for testing pre-fork behavior)  
#[derive(Clone, Debug)]
pub struct CrescendoParams {  
    /// Size of the sampled window for past median time calculation  
    pub past_median_time_sampled_window_size: u64,  
  
    /// Size of the sampled window for difficulty adjustment  
    pub sampled_difficulty_window_size: u64,  
  
    /// Target time per block in milliseconds (100ms for 10 BPS)  
    pub target_time_per_block: u64,  
  
    /// GHOSTDAG K parameter for 10 BPS (124)  
    pub ghostdag_k: KType,  
  
    /// Sample rate for past median time calculation (blocks between samples)  
    pub past_median_time_sample_rate: u64,  
  
    /// Sample rate for difficulty adjustment (blocks between samples)  
    pub difficulty_sample_rate: u64,  
  
    /// Maximum number of direct parents a block can have  
    pub max_block_parents: u8,  
  
    /// Maximum size of the GHOSTDAG mergeset  
    pub mergeset_size_limit: u64,  
  
    /// Merge depth bound for DAG structure  
    pub merge_depth: u64,  
  
    /// Finality depth (blocks until finality)  
    pub finality_depth: u64,  
  
    /// Pruning depth (blocks to keep before pruning)  
    pub pruning_depth: u64,  
  
    /// Maximum transaction inputs (reduced to 1000 for mass calculation efficiency)  
    pub max_tx_inputs: usize,  
  
    /// Maximum transaction outputs (reduced to 1000 for mass calculation efficiency)  
    pub max_tx_outputs: usize,  
  
    /// Maximum signature script length (10KB, limited by script engine)  
    pub max_signature_script_len: usize,  
  
    /// Maximum script public key length (10KB, limited by script engine)  
    pub max_script_public_key_len: usize,  
  
    /// Coinbase maturity period in blocks  
    pub coinbase_maturity: u64,  
}

/// The Crescendo hard fork parameters configured for 10 BPS.  
///  
/// All BPS-dependent values are calculated using [`TenBps`] constants.  
/// Transaction limits are set to control mass calculation costs while  
/// respecting script engine constraints.  
pub const CRESCENDO: CrescendoParams = CrescendoParams {
    past_median_time_sampled_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE,
    sampled_difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE,

    //
    // ~~~~~~~~~~~~~~~~~~ BPS dependent constants ~~~~~~~~~~~~~~~~~~
    //
    target_time_per_block: TenBps::target_time_per_block(),
    ghostdag_k: TenBps::ghostdag_k(),
    past_median_time_sample_rate: TenBps::past_median_time_sample_rate(),
    difficulty_sample_rate: TenBps::difficulty_adjustment_sample_rate(),
    max_block_parents: TenBps::max_block_parents(),
    mergeset_size_limit: TenBps::mergeset_size_limit(),
    merge_depth: TenBps::merge_depth_bound(),
    finality_depth: TenBps::finality_depth(),
    pruning_depth: TenBps::pruning_depth(),

    coinbase_maturity: TenBps::coinbase_maturity(),

    // Limit the cost of calculating compute/transient/storage masses
    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    // Transient mass enforces a limit of 125Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    max_signature_script_len: 10_000,
    // Compute mass enforces a limit of ~45.5Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    // Note that storage mass will kick in and gradually penalize also for lower lengths (generalized KIP-0009, plurality will be high).
    max_script_public_key_len: 10_000,
};

/// Network-specific consensus parameters.  
///  
/// Contains all consensus-sensitive settings and configurations. Changing any of these  
/// on a network node would prevent it from reaching consensus with unmodified nodes.  
///  
/// # Fork-Aware Parameters  
///  
/// Many parameters have "prior" versions that represent pre-Crescendo values. The actual  
/// active values are accessed through methods that return [`ForkedParam<T>`], which  
/// automatically provide the correct value based on the current DAA score and fork  
/// activation status.  
///  
/// # Parameter Categories  
///  
/// - **Network Identity**: `net`, `genesis`, `dns_seeders`  
/// - **GHOSTDAG**: `prior_ghostdag_k`, `prior_max_block_parents`, `prior_mergeset_size_limit`  
/// - **Difficulty Adjustment**: `prior_difficulty_window_size`, `min_difficulty_window_size`  
/// - **Timing**: `prior_target_time_per_block`, `timestamp_deviation_tolerance`  
/// - **Finality**: `prior_finality_depth`, `prior_pruning_depth`, `prior_merge_depth`  
/// - **Transactions**: `prior_max_tx_inputs`, `prior_max_tx_outputs`, mass parameters  
/// - **Coinbase**: `prior_coinbase_maturity`, `deflationary_phase_daa_score`  
/// - **Fork Management**: `crescendo`, `crescendo_activation`  
#[derive(Clone, Debug)]  
pub struct Params {  
    /// DNS seeders for peer discovery  
    pub dns_seeders: &'static [&'static str],  
      
    /// Network identifier (mainnet, testnet, devnet, simnet)  
    pub net: NetworkId,  
      
    /// Genesis block for this network  
    pub genesis: GenesisBlock,  
      
    /// GHOSTDAG K parameter before Crescendo fork  
    pub prior_ghostdag_k: KType,  
  
    /// Timestamp deviation tolerance in seconds  
    pub timestamp_deviation_tolerance: u64,  
  
    /// Target time per block in milliseconds (pre-Crescendo)  
    pub prior_target_time_per_block: u64,  
  
    /// Highest allowed proof of work difficulty value as a [`Uint256`]  
    pub max_difficulty_target: Uint256,  
  
    /// Highest allowed proof of work difficulty as a floating point number  
    pub max_difficulty_target_f64: f64,  
  
    /// Size of full blocks window for difficulty calculation (pre-Crescendo)  
    pub prior_difficulty_window_size: usize,  
  
    /// Minimum size a difficulty window must have to trigger DAA calculation  
    pub min_difficulty_window_size: usize,  
  
    /// Maximum number of direct parents a block can have (pre-Crescendo)  
    pub prior_max_block_parents: u8,  
      
    /// Maximum size of the GHOSTDAG mergeset (pre-Crescendo)  
    pub prior_mergeset_size_limit: u64,  
      
    /// Merge depth bound for DAG structure (pre-Crescendo)  
    pub prior_merge_depth: u64,  
      
    /// Finality depth in blocks (pre-Crescendo)  
    pub prior_finality_depth: u64,  
      
    /// Pruning depth in blocks (pre-Crescendo)  
    pub prior_pruning_depth: u64,  
  
    /// Maximum length of coinbase payload script public key  
    pub coinbase_payload_script_public_key_max_len: u8,  
      
    /// Maximum length of coinbase payload  
    pub max_coinbase_payload_len: usize,  
  
    /// Maximum transaction inputs (pre-Crescendo)  
    pub prior_max_tx_inputs: usize,  
      
    /// Maximum transaction outputs (pre-Crescendo)  
    pub prior_max_tx_outputs: usize,  
      
    /// Maximum signature script length (pre-Crescendo)  
    pub prior_max_signature_script_len: usize,  
      
    /// Maximum script public key length (pre-Crescendo)  
    pub prior_max_script_public_key_len: usize,  
  
    /// Mass per transaction byte  
    pub mass_per_tx_byte: u64,  
      
    /// Mass per script public key byte  
    pub mass_per_script_pub_key_byte: u64,  
      
    /// Mass per signature operation  
    pub mass_per_sig_op: u64,  
      
    /// Maximum block mass  
    pub max_block_mass: u64,  
  
    /// Parameter for scaling inverse KAS value to mass units (KIP-0009)  
    pub storage_mass_parameter: u64,  
  
    /// DAA score after which pre-deflationary period switches to deflationary period  
    pub deflationary_phase_daa_score: u64,  
  
    /// Base subsidy during pre-deflationary phase  
    pub pre_deflationary_phase_base_subsidy: u64,  
      
    /// Coinbase maturity period in blocks (pre-Crescendo)  
    pub prior_coinbase_maturity: u64,  
      
    /// Whether to skip proof of work validation (for testing)  
    pub skip_proof_of_work: bool,  
      
    /// Maximum block level for header chain  
    pub max_block_level: BlockLevel,  
      
    /// Pruning proof M parameter  
    pub pruning_proof_m: u64,  
  
    /// Crescendo hard fork parameters  
    pub crescendo: CrescendoParams,  
      
    /// Crescendo fork activation point  
    pub crescendo_activation: ForkActivation,  
}

impl Params {  
    /// Returns the size of the full blocks window for past median time calculation (pre-Crescendo).  
    ///  
    /// This is the legacy calculation used before the Crescendo fork, which inspects every block  
    /// in the window. The size is derived from the timestamp deviation tolerance.  
    ///  
    /// # Formula  
    ///  
    /// `window_size = 2 * timestamp_deviation_tolerance - 1`  
    #[inline]  
    #[must_use]  
    pub fn prior_past_median_time_window_size(&self) -> usize {  
        (2 * self.timestamp_deviation_tolerance - 1) as usize  
    }  
  
    /// Returns the size of the sampled blocks window for past median time calculation (post-Crescendo).  
    ///  
    /// After the Crescendo fork, the past median time calculation switches to a sampled window  
    /// for efficiency at higher block rates (10 BPS).  
    #[inline]  
    #[must_use]  
    pub fn sampled_past_median_time_window_size(&self) -> usize {  
        self.crescendo.past_median_time_sampled_window_size as usize  
    }  
  
    /// Returns the fork-aware past median time window size.  
    ///  
    /// This method returns a [`ForkedParam`] that automatically provides the correct window size  
    /// based on the current DAA score:  
    /// - Pre-Crescendo: Full window (every block sampled)  
    /// - Post-Crescendo: Sampled window (only specific blocks sampled)  
    #[inline]  
    #[must_use]  
    pub fn past_median_time_window_size(&self) -> ForkedParam<usize> {  
        ForkedParam::new(  
            self.prior_past_median_time_window_size(),  
            self.sampled_past_median_time_window_size(),  
            self.crescendo_activation,  
        )  
    }  
  
    /// Returns the fork-aware past median time sample rate.  
    ///  
    /// The sample rate determines how many blocks to skip between samples:  
    /// - Pre-Crescendo: 1 (every block is sampled)  
    /// - Post-Crescendo: Variable rate based on BPS (e.g., 10 for 10 BPS)  
    #[inline]  
    #[must_use]  
    pub fn past_median_time_sample_rate(&self) -> ForkedParam<u64> {  
        ForkedParam::new(1, self.crescendo.past_median_time_sample_rate, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware difficulty adjustment window size.  
    ///  
    /// This determines how many blocks are inspected for difficulty calculation:  
    /// - Pre-Crescendo: Full window (all blocks in range)  
    /// - Post-Crescendo: Sampled window (subset of blocks)  
    #[inline]  
    #[must_use]  
    pub fn difficulty_window_size(&self) -> ForkedParam<usize> {  
        ForkedParam::new(  
            self.prior_difficulty_window_size,  
            self.crescendo.sampled_difficulty_window_size as usize,  
            self.crescendo_activation,  
        )  
    }  
  
    /// Returns the fork-aware difficulty adjustment sample rate.  
    ///  
    /// The sample rate for difficulty window sampling:  
    /// - Pre-Crescendo: 1 (every block)  
    /// - Post-Crescendo: Variable rate based on BPS  
    #[inline]  
    #[must_use]  
    pub fn difficulty_sample_rate(&self) -> ForkedParam<u64> {  
        ForkedParam::new(1, self.crescendo.difficulty_sample_rate, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware target time per block in milliseconds.  
    ///  
    /// This is the target interval between blocks:  
    /// - Pre-Crescendo: 1000ms (1 BPS)  
    /// - Post-Crescendo: 100ms (10 BPS)  
    #[inline]  
    #[must_use]  
    pub fn target_time_per_block(&self) -> ForkedParam<u64> {  
        ForkedParam::new(self.prior_target_time_per_block, self.crescendo.target_time_per_block, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware blocks per second (BPS) rate.  
    ///  
    /// Calculated from the target time per block:  
    /// - Pre-Crescendo: 1 BPS (1000ms blocks)  
    /// - Post-Crescendo: 10 BPS (100ms blocks)  
    #[inline]  
    #[must_use]  
    pub fn bps(&self) -> ForkedParam<u64> {  
        ForkedParam::new(  
            1000 / self.prior_target_time_per_block,  
            1000 / self.crescendo.target_time_per_block,  
            self.crescendo_activation,  
        )  
    }
  
    /// Returns the fork-aware GHOSTDAG K parameter.  
    ///  
    /// K determines the maximum size of the anticone for blocks to be considered blue.  
    /// This is a critical parameter that affects the DAG's security and throughput:  
    /// - Pre-Crescendo: 18 (for 1 BPS)  
    /// - Post-Crescendo: 124 (for 10 BPS)  
    ///  
    /// The K value is calculated based on network delay and block rate to maintain  
    /// security guarantees. See [`calculate_ghostdag_k`] for the formula.  
    pub fn ghostdag_k(&self) -> ForkedParam<KType> {  
        ForkedParam::new(self.prior_ghostdag_k, self.crescendo.ghostdag_k, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware maximum number of direct parents a block can have.  
    ///  
    /// This limits the number of parent references in a block header:  
    /// - Pre-Crescendo: 10  
    /// - Post-Crescendo: Variable (typically K/2, capped at 16)  
    ///  
    /// The limit prevents quadratic growth in parent references as BPS increases  
    /// while ensuring sufficient DAG connectivity.  
    pub fn max_block_parents(&self) -> ForkedParam<u8> {  
        ForkedParam::new(self.prior_max_block_parents, self.crescendo.max_block_parents, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware mergeset size limit.  
    ///  
    /// The mergeset is the set of blocks merged by a block (its blue set minus  
    /// the blue set of its selected parent). This limit prevents excessive  
    /// storage complexity:  
    /// - Pre-Crescendo: 180 blocks  
    /// - Post-Crescendo: 248 blocks (2 * K, capped at 512)  
    pub fn mergeset_size_limit(&self) -> ForkedParam<u64> {  
        ForkedParam::new(self.prior_mergeset_size_limit, self.crescendo.mergeset_size_limit, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware merge depth bound.  
    ///  
    /// The merge depth is the maximum depth at which blocks can be merged without  
    /// violating bounded merge depth rules. Blocks deeper than this from the virtual  
    /// block cannot be merged:  
    /// - Pre-Crescendo: 3600 blocks  
    /// - Post-Crescendo: Scaled with BPS  
    pub fn merge_depth(&self) -> ForkedParam<u64> {  
        ForkedParam::new(self.prior_merge_depth, self.crescendo.merge_depth, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware finality depth.  
    ///  
    /// Blocks at this depth from the virtual block are considered final and  
    /// cannot be reorganized:  
    /// - Pre-Crescendo: 86400 blocks (~24 hours at 1 BPS)  
    /// - Post-Crescendo: Scaled with BPS to maintain similar time duration  
    pub fn finality_depth(&self) -> ForkedParam<u64> {  
        ForkedParam::new(self.prior_finality_depth, self.crescendo.finality_depth, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware pruning depth.  
    ///  
    /// Blocks deeper than this from the virtual block can be pruned from storage.  
    /// The pruning depth is calculated to ensure all finality guarantees are met:  
    /// - Pre-Crescendo: 185798 blocks  
    /// - Post-Crescendo: Calculated based on finality depth, merge depth, and K  
    ///  
    /// See the prunality paper for the mathematical derivation.  
    pub fn pruning_depth(&self) -> ForkedParam<u64> {  
        ForkedParam::new(self.prior_pruning_depth, self.crescendo.pruning_depth, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware coinbase maturity period.  
    ///  
    /// Coinbase outputs cannot be spent until this many blocks have been added  
    /// after the block containing the coinbase transaction:  
    /// - Pre-Crescendo: 100 blocks  
    /// - Post-Crescendo: Scaled with BPS  
    pub fn coinbase_maturity(&self) -> ForkedParam<u64> {  
        ForkedParam::new(self.prior_coinbase_maturity, self.crescendo.coinbase_maturity, self.crescendo_activation)  
    }
  
    /// Returns the fork-aware finality duration in milliseconds.  
    ///  
    /// This is the time duration corresponding to the finality depth:  
    /// - Pre-Crescendo: ~24 hours (86400 blocks * 1000ms)  
    /// - Post-Crescendo: ~24 hours (scaled to maintain similar duration)  
    pub fn finality_duration_in_milliseconds(&self) -> ForkedParam<u64> {  
        ForkedParam::new(  
            self.prior_target_time_per_block * self.prior_finality_depth,  
            self.crescendo.target_time_per_block * self.crescendo.finality_depth,  
            self.crescendo_activation,  
        )  
    }  
  
    /// Returns the fork-aware difficulty window duration in block units.  
    ///  
    /// This represents the span of blocks covered by the difficulty window:  
    /// - Pre-Crescendo: Full window size (2641 blocks)  
    /// - Post-Crescendo: Sample rate * window size (accounts for sampling)  
    pub fn difficulty_window_duration_in_block_units(&self) -> ForkedParam<u64> {  
        ForkedParam::new(  
            self.prior_difficulty_window_size as u64,  
            self.crescendo.difficulty_sample_rate * self.crescendo.sampled_difficulty_window_size,  
            self.crescendo_activation,  
        )  
    }  
  
    /// Returns the expected difficulty window duration in milliseconds.  
    ///  
    /// This is the expected time span covered by the difficulty adjustment window:  
    /// - Pre-Crescendo: ~44 minutes (2641 blocks * 1000ms)  
    /// - Post-Crescendo: Similar duration but with sampled blocks  
    pub fn expected_difficulty_window_duration_in_milliseconds(&self) -> ForkedParam<u64> {  
        ForkedParam::new(  
            self.prior_target_time_per_block * self.prior_difficulty_window_size as u64,  
            self.crescendo.target_time_per_block  
                * self.crescendo.difficulty_sample_rate  
                * self.crescendo.sampled_difficulty_window_size,  
            self.crescendo_activation,  
        )  
    }  
  
    /// Returns the depth at which the anticone of a chain block is final.  
    ///  
    /// This is the depth at which a block's anticone becomes a permanently closed set,  
    /// meaning no new blocks can be added to it. Based on the prunality analysis at  
    /// <https://github.com/kaspanet/docs/blob/main/Reference/prunality/Prunality.pdf>  
    ///  
    /// # Formula  
    ///  
    /// `anticone_finalization_depth = finality_depth + merge_depth +  
    ///  4 * mergeset_size_limit * ghostdag_k + 2 * ghostdag_k + 2`  
    ///  
    /// # Safety  
    ///  
    /// Returns the minimum of the calculated depth and pruning depth to avoid  
    /// situations where a block could be pruned before its anticone is finalized.  
    pub fn anticone_finalization_depth(&self) -> ForkedParam<u64> {  
        let prior_anticone_finalization_depth = self.prior_finality_depth  
            + self.prior_merge_depth  
            + 4 * self.prior_mergeset_size_limit * self.prior_ghostdag_k as u64  
            + 2 * self.prior_ghostdag_k as u64  
            + 2;  
  
        let new_anticone_finalization_depth = self.crescendo.finality_depth  
            + self.crescendo.merge_depth  
            + 4 * self.crescendo.mergeset_size_limit * self.crescendo.ghostdag_k as u64  
            + 2 * self.crescendo.ghostdag_k as u64  
            + 2;  
  
        // In mainnet it's guaranteed that `self.pruning_depth` is greater  
        // than `anticone_finalization_depth`, but for some tests we use  
        // a smaller (unsafe) pruning depth, so we return the minimum of  
        // the two to avoid a situation where a block can be pruned and  
        // not finalized.  
        ForkedParam::new(  
            min(self.prior_pruning_depth, prior_anticone_finalization_depth),  
            min(self.crescendo.pruning_depth, new_anticone_finalization_depth),  
            self.crescendo_activation,  
        )  
    }  
  
    /// Returns the fork-aware maximum transaction inputs.  
    ///  
    /// This limits the number of inputs a transaction can have:  
    /// - Pre-Crescendo: 1,000,000,000 (legacy, network-layer enforced)  
    /// - Post-Crescendo: 1,000 (to control mass calculation costs)  
    pub fn max_tx_inputs(&self) -> ForkedParam<usize> {  
        ForkedParam::new(self.prior_max_tx_inputs, self.crescendo.max_tx_inputs, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware maximum transaction outputs.  
    ///  
    /// This limits the number of outputs a transaction can have:  
    /// - Pre-Crescendo: 1,000,000,000 (legacy, network-layer enforced)  
    /// - Post-Crescendo: 1,000 (to control mass calculation costs)  
    pub fn max_tx_outputs(&self) -> ForkedParam<usize> {  
        ForkedParam::new(self.prior_max_tx_outputs, self.crescendo.max_tx_outputs, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware maximum signature script length.  
    ///  
    /// This limits the size of signature scripts in transaction inputs:  
    /// - Pre-Crescendo: 1,000,000,000 bytes (legacy, network-layer enforced)  
    /// - Post-Crescendo: 10,000 bytes (script engine limit)  
    pub fn max_signature_script_len(&self) -> ForkedParam<usize> {  
        ForkedParam::new(self.prior_max_signature_script_len, self.crescendo.max_signature_script_len, self.crescendo_activation)  
    }  
  
    /// Returns the fork-aware maximum script public key length.  
    ///  
    /// This limits the size of script public keys in transaction outputs:  
    /// - Pre-Crescendo: 1,000,000,000 bytes (legacy, network-layer enforced)  
    /// - Post-Crescendo: 10,000 bytes (script engine limit)  
    pub fn max_script_public_key_len(&self) -> ForkedParam<usize> {  
        ForkedParam::new(self.prior_max_script_public_key_len, self.crescendo.max_script_public_key_len, self.crescendo_activation)  
    }  
  
    /// Returns the network name with prefix (e.g., "kaspa-mainnet", "kaspa-testnet-10").  
    pub fn network_name(&self) -> String {  
        self.net.to_prefixed()  
    }  
  
    /// Returns the address prefix for this network.  
    pub fn prefix(&self) -> Prefix {  
        self.net.into()  
    }  
  
    /// Returns the default P2P port for this network.  
    pub fn default_p2p_port(&self) -> u16 {  
        self.net.default_p2p_port()  
    }  
  
    /// Returns the default RPC port for this network.  
    pub fn default_rpc_port(&self) -> u16 {  
        self.net.default_rpc_port()  
    }  
}

impl From<NetworkType> for Params {  
    /// Converts a network type to its corresponding consensus parameters.  
    ///  
    /// This provides a convenient way to get the default parameters for each network.  
    ///  
    /// # Examples  
    ///  
    /// ```  
    /// use kaspa_consensus_core::config::params::Params;  
    /// use kaspa_consensus_core::network::NetworkType;  
    ///  
    /// let mainnet_params: Params = NetworkType::Mainnet.into();  
    /// assert_eq!(mainnet_params.net.network_type, NetworkType::Mainnet);  
    /// ```  
    fn from(value: NetworkType) -> Self {  
        match value {  
            NetworkType::Mainnet => MAINNET_PARAMS,  
            NetworkType::Testnet => TESTNET_PARAMS,  
            NetworkType::Devnet => DEVNET_PARAMS,  
            NetworkType::Simnet => SIMNET_PARAMS,  
        }  
    }  
}  
  
impl From<NetworkId> for Params {  
    /// Converts a network ID to its corresponding consensus parameters.  
    ///  
    /// For testnet, this validates that the suffix is 10 (the only supported testnet version).  
    /// Other network types ignore the suffix.  
    ///  
    /// # Panics  
    ///  
    /// Panics if:  
    /// - Testnet is specified without a suffix  
    /// - Testnet is specified with a suffix other than 10  
    ///  
    /// # Examples  
    ///  
    /// ```  
    /// use kaspa_consensus_core::config::params::Params;  
    /// use kaspa_consensus_core::network::{NetworkId, NetworkType};  
    ///  
    /// let testnet_id = NetworkId::with_suffix(NetworkType::Testnet, 10);  
    /// let testnet_params: Params = testnet_id.into();  
    /// ```  
    fn from(value: NetworkId) -> Self {  
        match value.network_type {  
            NetworkType::Mainnet => MAINNET_PARAMS,  
            NetworkType::Testnet => match value.suffix {  
                Some(10) => TESTNET_PARAMS,  
                Some(x) => panic!("Testnet suffix {} is not supported", x),  
                None => panic!("Testnet suffix not provided"),  
            },  
            NetworkType::Devnet => DEVNET_PARAMS,  
            NetworkType::Simnet => SIMNET_PARAMS,  
        }  
    }  
}

/// Mainnet consensus parameters.  
///  
/// The production network configuration for Kaspa mainnet with the following characteristics:  
///  
/// # Network Configuration  
///  
/// - **Network Type**: Mainnet  
/// - **Block Rate**: 1 BPS (pre-Crescendo) → 10 BPS (post-Crescendo)  
/// - **GHOSTDAG K**: 18 (pre-Crescendo) → 124 (post-Crescendo)  
/// - **Crescendo Activation**: DAA score 110,165,000 (~May 5, 2025, 15:00 UTC)  
///  
/// # DNS Seeders  
///  
/// Mainnet uses multiple DNS seeders operated by community members for peer discovery.  
/// These seeders help new nodes bootstrap their connection to the network.  
///  
/// # Fork Activation  
///  
/// The Crescendo hard fork activates at DAA score 110,165,000, transitioning the network  
/// from 1 BPS to 10 BPS and updating various consensus parameters accordingly.  
///  
/// # Transaction Limits  
///  
/// Pre-Crescendo transaction limits are set to 1 billion (legacy, network-layer enforced).  
/// These will be reduced to more reasonable values in future hard forks.  
pub const MAINNET_PARAMS: Params = Params {  
    dns_seeders: &[  
        // This DNS seeder is run by Denis Mashkevich  
        "mainnet-dnsseed-1.kaspanet.org",  
        // This DNS seeder is run by Denis Mashkevich  
        "mainnet-dnsseed-2.kaspanet.org",  
        // This DNS seeder is run by Georges Künzli  
        "seeder1.kaspad.net",  
        // This DNS seeder is run by Georges Künzli  
        "seeder2.kaspad.net",  
        // This DNS seeder is run by Georges Künzli  
        "seeder3.kaspad.net",  
        // This DNS seeder is run by Georges Künzli  
        "seeder4.kaspad.net",  
        // This DNS seeder is run by Tim  
        "kaspadns.kaspacalc.net",  
        // This DNS seeder is run by supertypo  
        "n-mainnet.kaspa.ws",  
        // This DNS seeder is run by -gerri-  
        "dnsseeder-kaspa-mainnet.x-con.at",  
    ],  
    net: NetworkId::new(NetworkType::Mainnet),  
    genesis: GENESIS,  
    prior_ghostdag_k: LEGACY_DEFAULT_GHOSTDAG_K,  
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,  
    prior_target_time_per_block: 1000,  
    max_difficulty_target: MAX_DIFFICULTY_TARGET,  
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,  
    prior_difficulty_window_size: LEGACY_DIFFICULTY_WINDOW_SIZE,  
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,  
    prior_max_block_parents: 10,  
    prior_mergeset_size_limit: (LEGACY_DEFAULT_GHOSTDAG_K as u64) * 10,  
    prior_merge_depth: 3600,  
    prior_finality_depth: 86400,  
    prior_pruning_depth: 185798,  
    coinbase_payload_script_public_key_max_len: 150,  
    max_coinbase_payload_len: 204,  
  
    // This is technically a soft fork from the Go implementation since kaspad's consensus doesn't  
    // check these rules, but in practice it's enforced by the network layer that limits the message  
    // size to 1 GB.  
    // These values should be lowered to more reasonable amounts on the next planned HF/SF.  
    prior_max_tx_inputs: 1_000_000_000,  
    prior_max_tx_outputs: 1_000_000_000,  
    prior_max_signature_script_len: 1_000_000_000,  
    prior_max_script_public_key_len: 1_000_000_000,  
  
    mass_per_tx_byte: 1,  
    mass_per_script_pub_key_byte: 10,  
    mass_per_sig_op: 1000,  
    max_block_mass: 500_000,  
  
    storage_mass_parameter: STORAGE_MASS_PARAMETER,  
  
    // deflationary_phase_daa_score is the DAA score after which the pre-deflationary period  
    // switches to the deflationary period. This number is calculated as follows:  
    // We define a year as 365.25 days  
    // Half a year in seconds = 365.25 / 2 * 24 * 60 * 60 = 15778800  
    // The network was down for three days shortly after launch  
    // Three days in seconds = 3 * 24 * 60 * 60 = 259200  
    deflationary_phase_daa_score: 15778800 - 259200,  
    pre_deflationary_phase_base_subsidy: 50000000000,  
    prior_coinbase_maturity: 100,  
    skip_proof_of_work: false,  
    max_block_level: 225,  
    pruning_proof_m: 1000,  
  
    crescendo: CRESCENDO,  
    // Roughly 2025-05-05 1500 UTC  
    crescendo_activation: ForkActivation::new(110_165_000),  
};

/// Testnet 10 consensus parameters.  
///  
/// The test network configuration for Kaspa testnet-10 with the following characteristics:  
///  
/// # Network Configuration  
///  
/// - **Network Type**: Testnet with suffix 10  
/// - **Block Rate**: 1 BPS (pre-Crescendo) → 10 BPS (post-Crescendo)  
/// - **GHOSTDAG K**: 18 (pre-Crescendo) → 124 (post-Crescendo)  
/// - **Crescendo Activation**: DAA score 88,657,000 (~March 6, 2025, 18:30 UTC)  
///  
/// # Purpose  
///  
/// Testnet is used for testing protocol changes and new features before they are deployed  
/// to mainnet. The Crescendo fork activates earlier on testnet to allow for testing.  
///  
/// # DNS Seeders  
///  
/// Testnet uses dedicated DNS seeders for peer discovery on the test network.  
///  
/// # Differences from Mainnet  
///  
/// - Earlier Crescendo activation for testing  
/// - Separate genesis block  
/// - Different DNS seeders  
/// - Otherwise identical consensus parameters  
pub const TESTNET_PARAMS: Params = Params {  
    dns_seeders: &[  
        // This DNS seeder is run by Tiram  
        "seeder1-testnet.kaspad.net",  
        // This DNS seeder is run by -gerri-  
        "dnsseeder-kaspa-testnet.x-con.at",  
        // This DNS seeder is run by supertypo  
        "n-testnet-10.kaspa.ws",  
    ],  
    net: NetworkId::with_suffix(NetworkType::Testnet, 10),  
    genesis: TESTNET_GENESIS,  
    prior_ghostdag_k: LEGACY_DEFAULT_GHOSTDAG_K,  
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,  
    prior_target_time_per_block: 1000,  
    max_difficulty_target: MAX_DIFFICULTY_TARGET,  
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,  
    prior_difficulty_window_size: LEGACY_DIFFICULTY_WINDOW_SIZE,  
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,  
    prior_max_block_parents: 10,  
    prior_mergeset_size_limit: (LEGACY_DEFAULT_GHOSTDAG_K as u64) * 10,  
    prior_merge_depth: 3600,  
    prior_finality_depth: 86400,  
    prior_pruning_depth: 185798,  
    coinbase_payload_script_public_key_max_len: 150,  
    max_coinbase_payload_len: 204,  
  
    // This is technically a soft fork from the Go implementation since kaspad's consensus doesn't  
    // check these rules, but in practice it's enforced by the network layer that limits the message  
    // size to 1 GB.  
    // These values should be lowered to more reasonable amounts on the next planned HF/SF.  
    prior_max_tx_inputs: 1_000_000_000,  
    prior_max_tx_outputs: 1_000_000_000,  
    prior_max_signature_script_len: 1_000_000_000,  
    prior_max_script_public_key_len: 1_000_000_000,  
  
    mass_per_tx_byte: 1,  
    mass_per_script_pub_key_byte: 10,  
    mass_per_sig_op: 1000,  
    max_block_mass: 500_000,  
  
    storage_mass_parameter: STORAGE_MASS_PARAMETER,  
    // deflationary_phase_daa_score is the DAA score after which the pre-deflationary period  
    // switches to the deflationary period. This number is calculated as follows:  
    // We define a year as 365.25 days  
    // Half a year in seconds = 365.25 / 2 * 24 * 60 * 60 = 15778800  
    // The network was down for three days shortly after launch  
    // Three days in seconds = 3 * 24 * 60 * 60 = 259200  
    deflationary_phase_daa_score: 15778800 - 259200,  
    pre_deflationary_phase_base_subsidy: 50000000000,  
    prior_coinbase_maturity: 100,  
    skip_proof_of_work: false,  
    max_block_level: 250,  
    pruning_proof_m: 1000,  
  
    crescendo: CRESCENDO,  
    // 18:30 UTC, March 6, 2025  
    crescendo_activation: ForkActivation::new(88_657_000),  
};

/// Simnet (simulation network) consensus parameters.  
///  
/// A local testing network configured for 10 BPS operation with Crescendo parameters  
/// active from genesis. Designed for rapid development, testing, and benchmarking.  
///  
/// # Network Configuration  
///  
/// - **Network Type**: Simnet  
/// - **Block Rate**: 10 BPS (100ms blocks) from genesis  
/// - **GHOSTDAG K**: 18 (10 BPS configuration)  
/// - **Crescendo Activation**: Always active (from genesis)  
/// - **Proof of Work**: Disabled by default (`skip_proof_of_work: true`)  
///  
/// # Key Differences from Production Networks  
///  
/// - No DNS seeders (local network only)  
/// - Proof of work can be simulated  
/// - Crescendo parameters active from genesis  
/// - Reduced transaction limits (10,000 instead of 1 billion)  
/// - Allows at least 64 block parents to support mempool benchmarks  
///  
/// # Use Cases  
///  
/// - Local development and testing  
/// - Integration tests  
/// - Performance benchmarking  
/// - Protocol experimentation  
pub const SIMNET_PARAMS: Params = Params {  
    dns_seeders: &[],  
    net: NetworkId::new(NetworkType::Simnet),  
    genesis: SIMNET_GENESIS,  
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,  
    max_difficulty_target: MAX_DIFFICULTY_TARGET,  
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,  
    prior_difficulty_window_size: LEGACY_DIFFICULTY_WINDOW_SIZE,  
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,  
  
    //  
    // ~~~~~~~~~~~~~~~~~~ BPS dependent constants ~~~~~~~~~~~~~~~~~~  
    //  
    // Note we use a 10 BPS configuration for simnet  
    prior_ghostdag_k: TenBps::ghostdag_k(),  
    prior_target_time_per_block: TenBps::target_time_per_block(),  
    // For simnet, we deviate from TN11 configuration and allow at least 64 parents in order to support mempool benchmarks out of the box  
    prior_max_block_parents: if TenBps::max_block_parents() > 64 { TenBps::max_block_parents() } else { 64 },  
    prior_mergeset_size_limit: TenBps::mergeset_size_limit(),  
    prior_merge_depth: TenBps::merge_depth_bound(),  
    prior_finality_depth: TenBps::finality_depth(),  
    prior_pruning_depth: TenBps::pruning_depth(),  
    deflationary_phase_daa_score: TenBps::deflationary_phase_daa_score(),  
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),  
    prior_coinbase_maturity: TenBps::coinbase_maturity(),  
  
    coinbase_payload_script_public_key_max_len: 150,  
    max_coinbase_payload_len: 204,  
  
    prior_max_tx_inputs: 10_000,  
    prior_max_tx_outputs: 10_000,  
    prior_max_signature_script_len: 1_000_000,  
    prior_max_script_public_key_len: 1_000_000,  
  
    mass_per_tx_byte: 1,  
    mass_per_script_pub_key_byte: 10,  
    mass_per_sig_op: 1000,  
    max_block_mass: 500_000,  
  
    storage_mass_parameter: STORAGE_MASS_PARAMETER,  
  
    skip_proof_of_work: true, // For simnet only, PoW can be simulated by default  
    max_block_level: 250,  
    pruning_proof_m: PRUNING_PROOF_M,  
  
    crescendo: CRESCENDO,  
    crescendo_activation: ForkActivation::always(),  
};

/// Devnet (development network) consensus parameters.  
///  
/// A development network configured for 1 BPS operation with Crescendo parameters  
/// **not activated**. This allows testing pre-fork behavior and fork activation logic.  
///  
/// # Network Configuration  
///  
/// - **Network Type**: Devnet  
/// - **Block Rate**: 1 BPS (1000ms blocks)  
/// - **GHOSTDAG K**: 18 (legacy 1 BPS configuration)  
/// - **Crescendo Activation**: Never (disabled for testing pre-fork behavior)  
/// - **Proof of Work**: Enabled (unlike simnet)  
///  
/// # Key Differences from Other Networks  
///  
/// - No DNS seeders (local/private network)  
/// - Crescendo fork never activates (`ForkActivation::never()`)  
/// - Uses legacy 1 BPS parameters throughout  
/// - Proof of work validation enabled  
/// - Transaction limits set to 1 billion (legacy, network-layer enforced)  
///  
/// # Use Cases  
///  
/// - Testing pre-Crescendo consensus behavior  
/// - Fork activation testing (by changing `crescendo_activation`)  
/// - Private development networks  
/// - Integration testing with realistic parameters  
///  
/// # Deflationary Schedule  
///  
/// The deflationary phase begins at DAA score 15,519,600, calculated as:  
/// - Half year: 365.25 / 2 * 24 * 60 * 60 = 15,778,800 seconds  
/// - Minus 3-day network downtime: 3 * 24 * 60 * 60 = 259,200 seconds  
/// - Result: 15,778,800 - 259,200 = 15,519,600  
pub const DEVNET_PARAMS: Params = Params {  
    dns_seeders: &[],  
    net: NetworkId::new(NetworkType::Devnet),  
    genesis: DEVNET_GENESIS,  
    prior_ghostdag_k: LEGACY_DEFAULT_GHOSTDAG_K,  
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,  
    prior_target_time_per_block: 1000,  
    max_difficulty_target: MAX_DIFFICULTY_TARGET,  
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,  
    prior_difficulty_window_size: LEGACY_DIFFICULTY_WINDOW_SIZE,  
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,  
    prior_max_block_parents: 10,  
    prior_mergeset_size_limit: (LEGACY_DEFAULT_GHOSTDAG_K as u64) * 10,  
    prior_merge_depth: 3600,  
    prior_finality_depth: 86400,  
    prior_pruning_depth: 185798,  
    coinbase_payload_script_public_key_max_len: 150,  
    max_coinbase_payload_len: 204,  
  
    // This is technically a soft fork from the Go implementation since kaspad's consensus doesn't  
    // check these rules, but in practice it's enforced by the network layer that limits the message  
    // size to 1 GB.  
    // These values should be lowered to more reasonable amounts on the next planned HF/SF.  
    prior_max_tx_inputs: 1_000_000_000,  
    prior_max_tx_outputs: 1_000_000_000,  
    prior_max_signature_script_len: 1_000_000_000,  
    prior_max_script_public_key_len: 1_000_000_000,  
  
    mass_per_tx_byte: 1,  
    mass_per_script_pub_key_byte: 10,  
    mass_per_sig_op: 1000,  
    max_block_mass: 500_000,  
  
    storage_mass_parameter: STORAGE_MASS_PARAMETER,  
  
    // deflationary_phase_daa_score is the DAA score after which the pre-deflationary period  
    // switches to the deflationary period. This number is calculated as follows:  
    // We define a year as 365.25 days  
    // Half a year in seconds = 365.25 / 2 * 24 * 60 * 60 = 15778800  
    // The network was down for three days shortly after launch  
    // Three days in seconds = 3 * 24 * 60 * 60 = 259200  
    deflationary_phase_daa_score: 15778800 - 259200,  
    pre_deflationary_phase_base_subsidy: 50000000000,  
    prior_coinbase_maturity: 100,  
    skip_proof_of_work: false,  
    max_block_level: 250,  
    pruning_proof_m: 1000,  
  
    crescendo: CRESCENDO,  
    // TODO: Set this to always after the fork  
    crescendo_activation: ForkActivation::never(),  
};
