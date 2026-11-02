//! Pre-tokenization that reproduces the Python pattern exactly.
//!
//! The Python implementation runs
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+
//! ```
//!
//! through `re.findall`. Instead of a regex engine, this module walks the text
//! once and applies the same alternatives in the same order. Character classes
//! come from tables generated from Python's own `re` module, one per Unicode
//! version, so `\s`, `\d`, and the letter class agree with the interpreter on
//! every code point.

use crate::unicode_tables::{default_table, ClassTable};

/// The four classes that partition every Unicode scalar value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharClass {
    /// `\s`
    Space,
    /// `[^\W\d_]`: word characters that are neither decimal digits nor `_`.
    Letter,
    /// `\d`: decimal digits.
    Digit,
    /// `[^\s\w]|_`: everything else, including the underscore.
    Other,
}

/// Classify a character exactly as Python's `re` module does under the
/// Unicode version of `table`.
pub fn classify_with(c: char, table: ClassTable) -> CharClass {
    let cp = u32::from(c);
    if cp < 0x80 {
        return match c {
            '\t'..='\r' | '\x1c'..=' ' => CharClass::Space,
            'a'..='z' | 'A'..='Z' => CharClass::Letter,
            '0'..='9' => CharClass::Digit,
            _ => CharClass::Other,
        };
    }
    match table.binary_search_by(|&(lo, hi, _)| {
        if hi < cp {
            std::cmp::Ordering::Less
        } else if lo > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(index) => table[index].2,
        Err(_) => CharClass::Other,
    }
}

/// [`classify_with`] using the newest supported Unicode version.
pub fn classify(c: char) -> CharClass {
    classify_with(c, default_table())
}

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// Return the byte length of the match starting at `start`; `start` is a char
/// boundary strictly before the end of `text`.
fn match_at(text: &str, start: usize, table: ClassTable) -> usize {
    let classify = |c: char| classify_with(c, table);
    let rest = &text[start..];
    let mut chars = rest.char_indices();
    let (_, first) = chars.next().expect("start is before the end");

    if first == '\'' {
        if let Some(word) = CONTRACTIONS.iter().find(|w| rest.starts_with(**w)) {
            return word.len();
        }
    }

    // ` ?X+` for X in letters, digits, other: the optional space is taken
    // when it is followed by X, otherwise the run must start right here.
    let first_class = classify(first);
    let (run_start, run_class) = if first == ' ' {
        match rest[1..].chars().next() {
            Some(next) if classify(next) != CharClass::Space => (1, classify(next)),
            _ => (0, CharClass::Space),
        }
    } else {
        (0, first_class)
    };
    if run_class != CharClass::Space {
        let tail = &rest[run_start..];
        let run_len = tail
            .char_indices()
            .find(|&(_, c)| classify(c) != run_class)
            .map_or(tail.len(), |(i, _)| i);
        return run_start + run_len;
    }

    // `\s+(?!\S)` backtracks to leave the last whitespace character before a
    // non-space; when that leaves nothing, `\s+` takes the single character.
    let mut last_start = 0;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        if classify(c) != CharClass::Space {
            end = i;
            break;
        }
        last_start = i;
    }
    if end == rest.len() || last_start == 0 {
        end
    } else {
        last_start
    }
}

/// Split `text` into GPT-2 style chunks, identical to the Python `pretokenize`
/// running under the newest supported Unicode version.
pub fn pretokenize(text: &str) -> Vec<&str> {
    pretokenize_with(text, default_table())
}

/// [`pretokenize`] with the character classes of a specific Unicode version.
pub fn pretokenize_with(text: &str, table: ClassTable) -> Vec<&str> {
    let mut chunks = Vec::new();
    pretokenize_each(text, table, |chunk| chunks.push(chunk));
    chunks
}

/// Call `f` on each chunk without allocating a vector.
pub fn pretokenize_each<'a>(text: &'a str, table: ClassTable, mut f: impl FnMut(&'a str)) {
    let mut start = 0;
    while start < text.len() {
        let len = match_at(text, start, table);
        f(&text[start..start + len]);
        start += len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_leading_spaces_and_contractions() {
        assert_eq!(
            pretokenize("Hello world it's 42!"),
            vec!["Hello", " world", " it", "'s", " 42", "!"]
        );
    }

    #[test]
    fn whitespace_runs_leave_last_space_for_next_word() {
        assert_eq!(pretokenize("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(pretokenize("a \n\tb"), vec!["a", " \n", "\t", "b"]);
        assert_eq!(pretokenize("end   "), vec!["end", "   "]);
        assert_eq!(pretokenize("\nx"), vec!["\n", "x"]);
    }

    #[test]
    fn keeps_underscores() {
        assert_eq!(
            pretokenize("snake_case __x__"),
            vec!["snake", "_", "case", " __", "x", "__"]
        );
    }

    #[test]
    fn contraction_prefix_wins_over_punctuation_run() {
        assert_eq!(pretokenize("'sa"), vec!["'s", "a"]);
        assert_eq!(pretokenize("''s"), vec!["''", "s"]);
        assert_eq!(pretokenize("'x"), vec!["'", "x"]);
    }

    #[test]
    fn unicode_classes_follow_python_re() {
        assert_eq!(classify('\u{e9}'), CharClass::Letter);
        assert_eq!(classify('\u{b2}'), CharClass::Letter);
        assert_eq!(classify('\u{663}'), CharClass::Digit);
        assert_eq!(classify('\u{a0}'), CharClass::Space);
        assert_eq!(classify('\u{85}'), CharClass::Space);
        assert_eq!(classify('_'), CharClass::Other);
        assert_eq!(classify('\u{1F600}'), CharClass::Other);
    }

    #[test]
    fn tables_follow_their_unicode_version() {
        // U+1E030 (Cyrillic modifier letter) was added in Unicode 15.0.
        let old = crate::unicode_tables::table_for("14.0.0").unwrap();
        let new = crate::unicode_tables::table_for("15.0.0").unwrap();
        assert_eq!(classify_with('\u{1E030}', old), CharClass::Other);
        assert_eq!(classify_with('\u{1E030}', new), CharClass::Letter);
        assert!(crate::unicode_tables::table_for("1.0.0").is_none());
    }

    #[test]
    fn chunks_concatenate_to_input() {
        let text = "  caf\u{e9} na\u{ef}ve \u{1F600}\u{1F680} 123abc\t\n \u{65e5}\u{672c}_x  ";
        assert_eq!(pretokenize(text).concat(), text);
    }
}
