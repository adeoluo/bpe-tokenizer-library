//! Encoding and decoding with an ordered merge list.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

use crate::bytelevel::{glyphs_to_bytes, table};
use crate::pretokenize::pretokenize_each;
use crate::unicode_tables::{default_table, ClassTable};

/// Chunks with at least this many bytes use the heap-based merge loop; shorter
/// chunks use the direct scan, which is faster for them. Both pick the same
/// merge at every step, so the choice never changes the output.
const HEAP_THRESHOLD: usize = 64;

/// A token id that is not in the vocabulary, or that names a token which is
/// not made of byte-level glyphs (and so has no byte representation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(pub i64);

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown token id {}", self.0)
    }
}

impl std::error::Error for DecodeError {}

/// A byte-level BPE tokenizer built from an ordered merge list.
///
/// Tokens are identified by their glyph strings, as in Python, so the
/// vocabulary and ids are derived exactly as `BPETokenizer.from_merges` does.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    merges: Vec<(String, String)>,
    /// Glyph string of each id.
    tokens: Vec<String>,
    /// Raw bytes of each id, or `None` when the token contains non-glyph text.
    token_bytes: Vec<Option<Vec<u8>>>,
    /// Id of each single byte.
    byte_ids: [u32; 256],
    /// `(left id, right id) -> (rank, merged id)`.
    ranks: HashMap<(u32, u32), (u32, u32)>,
    /// Character classes for pre-tokenization.
    classes: ClassTable,
}

impl Tokenizer {
    /// Build a tokenizer from merges, mirroring `BPETokenizer.from_merges`,
    /// that pre-tokenizes with the newest supported Unicode version.
    pub fn from_merges(merges: Vec<(String, String)>) -> Self {
        Self::from_merges_with(merges, default_table())
    }

    /// [`Self::from_merges`] with the character classes of a specific Unicode
    /// version (see [`crate::unicode_tables::table_for`]).
    pub fn from_merges_with(merges: Vec<(String, String)>, classes: ClassTable) -> Self {
        let byte_level = table();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut tokens: Vec<String> = Vec::new();
        let mut intern = |token: String, vocab: &mut HashMap<String, u32>| -> u32 {
            if let Some(&id) = vocab.get(&token) {
                return id;
            }
            let id = tokens.len() as u32;
            vocab.insert(token.clone(), id);
            tokens.push(token);
            id
        };
        for &byte in &byte_level.order {
            intern(
                byte_level.glyph_of_byte[byte as usize].to_string(),
                &mut vocab,
            );
        }
        for (a, b) in &merges {
            intern(format!("{a}{b}"), &mut vocab);
        }

        // Python builds `{pair: index for index, pair in enumerate(merges)}`,
        // so a repeated pair keeps its last rank. Pairs whose sides are not in
        // the vocabulary can never occur during encoding and are skipped.
        let mut ranks = HashMap::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            if let (Some(&left), Some(&right)) = (vocab.get(a), vocab.get(b)) {
                let merged = vocab[&format!("{a}{b}")];
                ranks.insert((left, right), (rank as u32, merged));
            }
        }

        let mut byte_ids = [0u32; 256];
        for byte in 0..256usize {
            byte_ids[byte] = vocab[&byte_level.glyph_of_byte[byte].to_string()];
        }
        let token_bytes = tokens.iter().map(|t| glyphs_to_bytes(t)).collect();
        Tokenizer {
            merges,
            tokens,
            token_bytes,
            byte_ids,
            ranks,
            classes,
        }
    }

    /// Train merges on `corpus` and build a tokenizer from them.
    pub fn train(corpus: &[&str], vocab_size: i64) -> Self {
        Self::from_merges(crate::trainer::train_bpe(corpus, vocab_size))
    }

    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Glyph string of a token id.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    fn rank(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        self.ranks.get(&(left, right)).copied()
    }

    /// Merge loop of Python's `_bpe`: repeatedly merge the leftmost occurrence
    /// of the lowest-ranked adjacent pair.
    fn bpe_scan(&self, symbols: &mut Vec<u32>) {
        while symbols.len() >= 2 {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some((rank, merged)) = self.rank(symbols[i], symbols[i + 1]) {
                    if best.is_none_or(|(best_rank, _, _)| rank < best_rank) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            let Some((_, index, merged)) = best else {
                break;
            };
            symbols[index] = merged;
            symbols.remove(index + 1);
        }
    }

    /// The same merge order as [`Self::bpe_scan`] in `O(n log n)`: a min-heap
    /// keyed by `(rank, position)` over a linked list of symbols. Positions are
    /// original indices, so their order is the left-to-right order.
    fn bpe_heap(&self, symbols: &mut Vec<u32>) {
        let n = symbols.len();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some((rank, _)) = self.rank(symbols[i], symbols[i + 1]) {
                heap.push(Reverse((rank, i)));
            }
        }
        while let Some(Reverse((rank, left))) = heap.pop() {
            if !alive[left] || next[left] >= n {
                continue;
            }
            let right = next[left];
            let Some((current, merged)) = self.rank(symbols[left], symbols[right]) else {
                continue;
            };
            if current != rank {
                continue;
            }
            symbols[left] = merged;
            alive[right] = false;
            next[left] = next[right];
            if next[left] < n {
                prev[next[left]] = left;
            }
            if let Some(p) = prev.get(left).copied().filter(|&p| p < n) {
                if let Some((r, _)) = self.rank(symbols[p], symbols[left]) {
                    heap.push(Reverse((r, p)));
                }
            }
            if next[left] < n {
                if let Some((r, _)) = self.rank(symbols[left], symbols[next[left]]) {
                    heap.push(Reverse((r, left)));
                }
            }
        }
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            out.push(symbols[i]);
            i = next[i];
        }
        *symbols = out;
    }

    fn encode_chunk(&self, chunk: &[u8], ids: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = chunk.iter().map(|&b| self.byte_ids[b as usize]).collect();
        if symbols.len() >= HEAP_THRESHOLD {
            self.bpe_heap(&mut symbols);
        } else {
            self.bpe_scan(&mut symbols);
        }
        ids.extend_from_slice(&symbols);
    }

    /// Encode `text` into token ids, identical to `BPETokenizer.encode`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        pretokenize_each(text, self.classes, |chunk| {
            self.encode_chunk(chunk.as_bytes(), &mut ids)
        });
        ids
    }

    /// Decode token ids, identical to `BPETokenizer.decode`: invalid UTF-8 is
    /// replaced with U+FFFD per maximal invalid subsequence, as Python's
    /// `errors="replace"` does.
    pub fn decode(&self, ids: &[i64]) -> Result<String, DecodeError> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }

    /// Concatenated raw bytes of the tokens.
    pub fn decode_bytes(&self, ids: &[i64]) -> Result<Vec<u8>, DecodeError> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let bytes = usize::try_from(id)
                .ok()
                .and_then(|i| self.token_bytes.get(i))
                .and_then(Option::as_ref)
                .ok_or(DecodeError(id))?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(a: &str, b: &str) -> (String, String) {
        (a.to_string(), b.to_string())
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    #[test]
    fn base_vocabulary_matches_python_order() {
        let tok = Tokenizer::from_merges(vec![]);
        assert_eq!(tok.vocab_size(), 256);
        assert_eq!(tok.encode("!"), vec![0]);
        assert_eq!(tok.encode("a"), vec![u32::from(b'a' - b'!')]);
        assert_eq!(tok.encode(" "), vec![220]);
    }

    #[test]
    fn applies_merges_by_rank() {
        let tok = Tokenizer::from_merges(vec![
            pair("h", "e"),
            pair("l", "l"),
            pair("he", "ll"),
            pair("\u{120}", "w"),
        ]);
        let ids = tok.encode("hello wow");
        let tokens: Vec<&str> = ids.iter().map(|&i| tok.token(i).unwrap()).collect();
        assert_eq!(tokens, vec!["hell", "o", "\u{120}w", "o", "w"]);
        assert_eq!(
            tok.decode(&ids.iter().map(|&i| i64::from(i)).collect::<Vec<_>>())
                .unwrap(),
            "hello wow"
        );
    }

    #[test]
    fn repeated_pair_keeps_last_rank_and_dedupes_vocab() {
        let tok = Tokenizer::from_merges(vec![pair("a", "b"), pair("b", "c"), pair("a", "b")]);
        assert_eq!(tok.vocab_size(), 258);
        // (a, b) has rank 2, so (b, c) at rank 1 merges first.
        let tokens: Vec<&str> = tok
            .encode("abc")
            .iter()
            .map(|&i| tok.token(i).unwrap())
            .collect();
        assert_eq!(tokens, vec!["a", "bc"]);
    }

    #[test]
    fn decode_rejects_unknown_ids_and_replaces_invalid_utf8() {
        let tok = Tokenizer::from_merges(vec![]);
        assert_eq!(tok.decode(&[-1]), Err(DecodeError(-1)));
        assert_eq!(tok.decode(&[256]), Err(DecodeError(256)));
        let id_of = |b: u8| i64::from(tok.encode_bytes_for_test(b));
        assert_eq!(
            tok.decode(&[id_of(0xFF), id_of(b'a')]).unwrap(),
            "\u{FFFD}a"
        );
    }

    impl Tokenizer {
        fn encode_bytes_for_test(&self, byte: u8) -> u32 {
            self.byte_ids[byte as usize]
        }
    }

    #[test]
    fn heap_and_scan_agree_on_arbitrary_merges() {
        let mut rng = Lcg(7);
        let alphabet = ["a", "b", "c", "d"];
        let mut tokens: Vec<String> = alphabet.iter().map(|s| s.to_string()).collect();
        let mut merges = Vec::new();
        for _ in 0..60 {
            let a = tokens[(rng.next() as usize) % tokens.len()].clone();
            let b = tokens[(rng.next() as usize) % tokens.len()].clone();
            tokens.push(format!("{a}{b}"));
            merges.push((a, b));
        }
        let tok = Tokenizer::from_merges(merges);
        for _ in 0..300 {
            let len = (rng.next() % 200) as usize;
            let text: Vec<u8> = (0..len)
                .map(|_| b"abcd"[(rng.next() % 4) as usize])
                .collect();
            let mut scan: Vec<u32> = text.iter().map(|&b| tok.byte_ids[b as usize]).collect();
            let mut heap = scan.clone();
            tok.bpe_scan(&mut scan);
            if heap.len() >= 2 {
                tok.bpe_heap(&mut heap);
            }
            assert_eq!(scan, heap);
        }
    }
}
