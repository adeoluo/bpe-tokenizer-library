//! GPT-2's reversible map between the 256 byte values and printable glyphs.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The glyph for each byte value, plus the order in which Python's
/// `bytes_to_unicode()` dict yields its entries (printable bytes first, then
/// the remaining bytes in increasing order).
pub struct ByteLevel {
    pub glyph_of_byte: [char; 256],
    pub byte_of_glyph: HashMap<char, u8>,
    pub order: [u8; 256],
}

fn is_printable(byte: u32) -> bool {
    (u32::from('!')..=u32::from('~')).contains(&byte)
        || (0xA1..=0xAC).contains(&byte)
        || (0xAE..=0xFF).contains(&byte)
}

fn build() -> ByteLevel {
    let mut glyph_of_byte = ['\0'; 256];
    let mut order = [0u8; 256];
    let mut position = 0;
    for byte in 0..256u32 {
        if is_printable(byte) {
            glyph_of_byte[byte as usize] = char::from_u32(byte).expect("printable byte");
            order[position] = byte as u8;
            position += 1;
        }
    }
    let mut extra = 0;
    for byte in 0..256u32 {
        if !is_printable(byte) {
            glyph_of_byte[byte as usize] = char::from_u32(256 + extra).expect("valid glyph");
            order[position] = byte as u8;
            position += 1;
            extra += 1;
        }
    }
    let byte_of_glyph = (0..256usize)
        .map(|byte| (glyph_of_byte[byte], byte as u8))
        .collect();
    ByteLevel {
        glyph_of_byte,
        byte_of_glyph,
        order,
    }
}

/// Return the process-wide byte-level table.
pub fn table() -> &'static ByteLevel {
    static TABLE: OnceLock<ByteLevel> = OnceLock::new();
    TABLE.get_or_init(build)
}

/// Map raw bytes to their glyph string (Python's `encode_bytes` on UTF-8 bytes).
pub fn bytes_to_glyphs(bytes: &[u8]) -> String {
    let table = table();
    bytes
        .iter()
        .map(|&b| table.glyph_of_byte[b as usize])
        .collect()
}

/// Map a glyph string back to raw bytes, or `None` if it contains a character
/// that is not one of the 256 glyphs.
pub fn glyphs_to_bytes(glyphs: &str) -> Option<Vec<u8>> {
    let table = table();
    glyphs
        .chars()
        .map(|c| table.byte_of_glyph.get(&c).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_a_bijection() {
        let table = table();
        assert_eq!(table.byte_of_glyph.len(), 256);
        for byte in 0..=255u8 {
            let glyph = table.glyph_of_byte[byte as usize];
            assert_eq!(table.byte_of_glyph[&glyph], byte);
        }
    }

    #[test]
    fn known_glyphs_match_gpt2() {
        assert_eq!(bytes_to_glyphs(b" "), "\u{120}");
        assert_eq!(bytes_to_glyphs(b"\n"), "\u{10A}");
        assert_eq!(bytes_to_glyphs(b"a!"), "a!");
        assert_eq!(table().order[0], b'!');
        assert_eq!(table().order[255], 0xAD);
    }

    #[test]
    fn round_trips_utf8() {
        let text = "caf\u{e9} \u{1F600}";
        let glyphs = bytes_to_glyphs(text.as_bytes());
        assert_eq!(glyphs_to_bytes(&glyphs).unwrap(), text.as_bytes());
        assert_eq!(glyphs_to_bytes(" "), None);
    }
}
