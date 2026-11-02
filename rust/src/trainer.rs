//! Training merges, identical to the Python `train_bpe`.
//!
//! Python recounts every adjacent pair after each merge. This version keeps
//! the counts up to date incrementally, touching only the words that contain
//! the merged pair, which yields the same counts and therefore the same merge
//! at every step: the most frequent pair, ties broken by the lexicographically
//! smallest `(left, right)` glyph strings.

use std::collections::{HashMap, HashSet};

use crate::pretokenize::pretokenize_each;
use crate::unicode_tables::{default_table, ClassTable};

type Pair = (u32, u32);

fn add_pairs(
    word: &[u32],
    index: usize,
    freq: i64,
    counts: &mut HashMap<Pair, i64>,
    where_: &mut HashMap<Pair, HashSet<usize>>,
) {
    for w in word.windows(2) {
        let pair = (w[0], w[1]);
        *counts.entry(pair).or_insert(0) += freq;
        where_.entry(pair).or_default().insert(index);
    }
}

fn remove_pairs(word: &[u32], freq: i64, counts: &mut HashMap<Pair, i64>) {
    for w in word.windows(2) {
        let pair = (w[0], w[1]);
        if let Some(count) = counts.get_mut(&pair) {
            *count -= freq;
            if *count <= 0 {
                counts.remove(&pair);
            }
        }
    }
}

fn merge_word(word: &[u32], pair: Pair, merged: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
            out.push(merged);
            i += 2;
        } else {
            out.push(word[i]);
            i += 1;
        }
    }
    out
}

/// Return the ordered merges learned from `corpus` for a target vocabulary of
/// `vocab_size` (256 base glyphs plus the merges), pre-tokenizing with the
/// newest supported Unicode version.
pub fn train_bpe<S: AsRef<str>>(corpus: &[S], vocab_size: i64) -> Vec<(String, String)> {
    train_bpe_with(corpus, vocab_size, default_table())
}

/// [`train_bpe`] with the character classes of a specific Unicode version.
pub fn train_bpe_with<S: AsRef<str>>(
    corpus: &[S],
    vocab_size: i64,
    table: ClassTable,
) -> Vec<(String, String)> {
    let target = usize::try_from(vocab_size.saturating_sub(256)).unwrap_or(0);
    let glyphs = crate::bytelevel::table();

    let mut names: Vec<String> = Vec::new();
    let mut ids: HashMap<String, u32> = HashMap::new();
    for byte in 0..256usize {
        let glyph = glyphs.glyph_of_byte[byte].to_string();
        ids.insert(glyph.clone(), byte as u32);
        names.push(glyph);
    }

    let mut frequencies: HashMap<&str, i64> = HashMap::new();
    for document in corpus {
        pretokenize_each(document.as_ref(), table, |chunk| {
            *frequencies.entry(chunk).or_insert(0) += 1;
        });
    }
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(frequencies.len());
    let mut freqs: Vec<i64> = Vec::with_capacity(frequencies.len());
    for (chunk, freq) in frequencies {
        words.push(chunk.bytes().map(u32::from).collect());
        freqs.push(freq);
    }

    let mut counts: HashMap<Pair, i64> = HashMap::new();
    let mut where_: HashMap<Pair, HashSet<usize>> = HashMap::new();
    for (index, word) in words.iter().enumerate() {
        add_pairs(word, index, freqs[index], &mut counts, &mut where_);
    }

    let mut merges = Vec::new();
    while merges.len() < target {
        let best = counts.iter().max_by(|(pa, ca), (pb, cb)| {
            ca.cmp(cb).then_with(|| {
                let a = (names[pa.0 as usize].as_str(), names[pa.1 as usize].as_str());
                let b = (names[pb.0 as usize].as_str(), names[pb.1 as usize].as_str());
                b.cmp(&a)
            })
        });
        let Some((&pair, _)) = best else {
            break;
        };
        let name = format!("{}{}", names[pair.0 as usize], names[pair.1 as usize]);
        let merged = *ids.entry(name.clone()).or_insert_with(|| {
            names.push(name);
            (names.len() - 1) as u32
        });
        let affected = where_.remove(&pair).unwrap_or_default();
        for index in affected {
            let word = &words[index];
            if !word.windows(2).any(|w| (w[0], w[1]) == pair) {
                continue;
            }
            let updated = merge_word(word, pair, merged);
            remove_pairs(word, freqs[index], &mut counts);
            add_pairs(&updated, index, freqs[index], &mut counts, &mut where_);
            words[index] = updated;
        }
        merges.push((
            names[pair.0 as usize].clone(),
            names[pair.1 as usize].clone(),
        ));
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_most_frequent_pair_first() {
        let merges = train_bpe(&["aaab", "aaab", "ab"], 258);
        assert_eq!(
            merges,
            vec![("a".into(), "a".into()), ("a".into(), "b".into())]
        );
    }

    #[test]
    fn ties_break_on_glyph_strings_not_bytes() {
        // " a" and "ab" each occur once; the space glyph U+0120 sorts after "a".
        let merges = train_bpe(&["ab a"], 257);
        assert_eq!(merges, vec![("a".into(), "b".into())]);
    }

    #[test]
    fn stops_when_no_pairs_remain() {
        assert!(train_bpe(&["a", "b", " "], 1000).is_empty());
        assert!(train_bpe::<&str>(&[], 1000).is_empty());
        assert!(train_bpe(&["abc"], -5).is_empty());
    }

    #[test]
    fn deterministic() {
        let corpus = ["the quick brown fox", "the lazy dog", "the end"];
        assert_eq!(train_bpe(&corpus, 300), train_bpe(&corpus, 300));
    }
}
