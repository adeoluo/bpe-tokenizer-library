//! Rust core of `bpe-tokenizer-library`.
//!
//! A byte-level byte-pair-encoding tokenizer implementing the GPT-2 algorithm.
//! Every function here mirrors the reference Python implementation in
//! `src/bpe_tokenizer_library/` and is required to produce byte-for-byte
//! identical output for the same merges: the same pre-tokenization chunks, the
//! same token ids, the same decoded text, and the same trained merge list.

pub mod bytelevel;
pub mod pretokenize;
pub mod tokenizer;
pub mod trainer;
pub mod unicode_tables;

#[cfg(feature = "python")]
mod python;

pub use pretokenize::{pretokenize, pretokenize_with};
pub use tokenizer::{DecodeError, Tokenizer};
pub use trainer::{train_bpe, train_bpe_with};
pub use unicode_tables::{UNICODE_VERSION, table_for};
