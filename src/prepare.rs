//! Source preparation: a port of upstream `prepare-tree.js` and
//! `tree-features.js` (gpu-lexer 0.0.6, tokenizer v2, feature v3).
//!
//! One scan splits UTF-16 code units into word runs, horizontal-space runs,
//! newlines (CRLF is one) and single symbols, and packs each part into two
//! `u32` words exactly as the upstream runtime does, so the GPU shader can
//! read them unchanged and the CPU forward pass sees identical features.

/// A run of ASCII letters, digits, `_` and non-ASCII code units.
pub const PART_WORD: u32 = 0;
/// A run of spaces, tabs, vertical tabs and form feeds.
pub const PART_SPACE: u32 = 1;
/// `\n`, `\r` or `\r\n`.
pub const PART_NEWLINE: u32 = 2;
/// Any other single code unit.
pub const PART_SYMBOL: u32 = 3;

const fn ascii_kind(code: u32) -> u32 {
    match code {
        10 | 13 => PART_NEWLINE,
        9 | 11 | 12 | 32 => PART_SPACE,
        48..=57 | 65..=90 | 95 | 97..=122 => PART_WORD,
        _ => PART_SYMBOL,
    }
}

const fn word_flags(code: u32) -> u32 {
    match code {
        97..=122 => 1,
        65..=90 => 2,
        48..=57 => 4,
        95 => 8,
        _ => 0,
    }
}

static ASCII_KIND: [u8; 128] = {
    let mut table = [0u8; 128];
    let mut code = 0;
    while code < 128 {
        table[code] = ascii_kind(code as u32) as u8;
        code += 1;
    }
    table
};

static WORD_FLAGS: [u8; 128] = {
    let mut table = [0u8; 128];
    let mut code = 0;
    while code < 128 {
        table[code] = word_flags(code as u32) as u8;
        code += 1;
    }
    table
};

// Pair codes 1-14 in order; 15 is never emitted.
const PAIRS: &[u8] = b"///**/--=>::</{{${}}->???.<>";

static PAIR_CODES: [u8; 128 * 128] = {
    let mut table = [0u8; 128 * 128];
    let mut index = 0;
    while index < PAIRS.len() {
        table[((PAIRS[index] as usize) << 7) | PAIRS[index + 1] as usize] = (index / 2 + 1) as u8;
        index += 2;
    }
    table
};

// Zero is the edge/no-neighbor sentinel; 31 buckets cover every boundary.
static SYMBOL_PAIR_HASHES: [u8; 128 * 128] = {
    let mut table = [0u8; 128 * 128];
    let mut code = 0u32;
    while code < 128 {
        let mut right = 0u32;
        while right < 128 {
            table[((code << 7) | right) as usize] = (1 + (((code + 1).wrapping_mul(131)) ^ right) % 31) as u8;
            right += 1;
        }
        code += 1;
    }
    table
};

/// A prepared source: two packed words per part and each part's UTF-16 range.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Prepared {
    /// `[packed, context]` per part, the layout the shader reads.
    pub data: Vec<u32>,
    /// `[start, end)` per part, in UTF-16 code units.
    pub ranges: Vec<u32>,
}

impl Prepared {
    /// How many parts the source split into.
    pub fn len(&self) -> usize {
        self.ranges.len() / 2
    }

    /// Whether the source was empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Part `index`'s kind (`PART_*`).
    pub fn kind(&self, index: usize) -> u32 {
        self.data[index * 2] & 3
    }

    /// Whether part `index` is whitespace, which the model never classifies.
    pub fn is_space(&self, index: usize) -> bool {
        let kind = self.kind(index);
        kind == PART_SPACE || kind == PART_NEWLINE
    }
}

/// Splits `source` (UTF-16 code units) into parts, as `prepareTreeSource`.
pub fn prepare(source: &[u16]) -> Prepared {
    let length = source.len();
    // A source never has more parts than code units; most have far fewer.
    let estimate = length / 3 + 16;
    let mut data: Vec<u32> = Vec::with_capacity(estimate * 2);
    let mut ranges: Vec<u32> = Vec::with_capacity(estimate * 2);
    let mut offset = 0usize;
    let mut line_start = true;
    let mut previous_last = 0u32;
    let mut previous_kind = u32::MAX;
    while offset < length {
        let from = offset;
        let first_raw = u32::from(source[offset]);
        offset += 1;
        let first = if first_raw > 127 { 95 } else { first_raw };
        let kind = if first_raw > 127 { PART_WORD } else { u32::from(ASCII_KIND[first_raw as usize]) };
        let mut flags: u32 = if line_start { 16 } else { 0 };
        let mut last = first;
        let mut packed = kind | (first << 5);
        if kind == PART_WORD {
            let mut hash = 2_166_136_261u32 ^ first;
            hash = hash.wrapping_mul(16_777_619);
            let mut secondary = 0x9e37_79b9u32 ^ first;
            secondary = secondary.wrapping_mul(2_246_822_519);
            flags |= u32::from(WORD_FLAGS[first as usize]);
            while offset < length {
                let raw = u32::from(source[offset]);
                let mut normalized = 95;
                if raw <= 127 {
                    if u32::from(ASCII_KIND[raw as usize]) != PART_WORD {
                        break;
                    }
                    normalized = raw;
                }
                hash = (hash ^ normalized).wrapping_mul(16_777_619);
                secondary = (secondary ^ normalized).wrapping_mul(2_246_822_519);
                flags |= u32::from(WORD_FLAGS[normalized as usize]);
                last = normalized;
                offset += 1;
            }
            packed |= (hash & 255) << 19;
            flags |= (secondary & 127) << 15;
        } else if kind == PART_SPACE {
            if first_raw == 9 {
                flags |= 32;
            }
            while offset < length {
                let raw = u32::from(source[offset]);
                if raw > 127 || u32::from(ASCII_KIND[raw as usize]) != PART_SPACE {
                    break;
                }
                if raw == 9 {
                    flags |= 32;
                }
                last = raw;
                offset += 1;
            }
        } else if kind == PART_NEWLINE {
            if first_raw == 13 && offset < length && source[offset] == 10 {
                last = 10;
                offset += 1;
            }
            line_start = true;
        } else if first_raw == 92 {
            flags |= 64;
        }
        if kind == PART_WORD || kind == PART_SYMBOL {
            line_start = false;
        }

        let part_length = offset - from;
        let bucket = if part_length > 127 { 7 } else { 31 - (part_length as u32).leading_zeros() };
        packed |= (bucket << 2) | (last << 12);
        if !ranges.is_empty() {
            let pair_index = ((previous_last << 7) | first) as usize;
            let pair = u32::from(PAIR_CODES[pair_index]);
            let mut previous_context = pair << 11;
            flags |= pair << 7;
            if kind == PART_SYMBOL || previous_kind == PART_SYMBOL {
                let generic = u32::from(SYMBOL_PAIR_HASHES[pair_index]);
                flags |= generic << 22;
                previous_context |= generic << 27;
            }
            let previous = data.len() - 1;
            data[previous] |= previous_context;
        }
        data.push(packed);
        data.push(flags);
        ranges.push(from as u32);
        ranges.push(offset as u32);
        previous_last = last;
        previous_kind = kind;
    }
    Prepared { data, ranges }
}

/// Row offsets of the feature-v3 layout with 256 lexeme hash buckets
/// (`treeFeatureLayout(256, 3)`).
#[allow(missing_docs)]
pub mod layout {
    pub const PRIMARY_HASH: usize = 268;
    pub const SECONDARY_HASH: usize = PRIMARY_HASH + 256;
    pub const FLAGS: usize = SECONDARY_HASH + 128;
    pub const PREVIOUS_PAIR: usize = FLAGS + 7;
    pub const NEXT_PAIR: usize = PREVIOUS_PAIR + 16;
    pub const PREVIOUS_SYMBOL_PAIR: usize = NEXT_PAIR + 16;
    pub const NEXT_SYMBOL_PAIR: usize = PREVIOUS_SYMBOL_PAIR + 32;
    pub const INPUT_SIZE: usize = NEXT_SYMBOL_PAIR + 32;
}

/// The sparse feature rows of part `index`, in the order upstream
/// `treeFeatureIndices` lists them (the order the embedding sums them in).
/// Returns how many of `out` were written (at most 17).
pub fn feature_indices(data: &[u32], index: usize, out: &mut [u16; 17]) -> usize {
    let packed = data[index * 2];
    let context = data[index * 2 + 1];
    let kind = packed & 3;
    let mut count = 0;
    let mut push = |row: usize| {
        out[count] = row as u16;
        count += 1;
    };
    push(kind as usize);
    push(4 + ((packed >> 2) & 7) as usize);
    push(12 + ((packed >> 5) & 127) as usize);
    push(140 + ((packed >> 12) & 127) as usize);
    if kind == PART_WORD {
        push(layout::PRIMARY_HASH + ((packed >> 19) & 255) as usize);
        push(layout::SECONDARY_HASH + ((context >> 15) & 127) as usize);
    }
    let flags = context & 127;
    for bit in 0..7 {
        if flags & (1 << bit) != 0 {
            push(layout::FLAGS + bit);
        }
    }
    let previous_pair = ((context >> 7) & 15) as usize;
    let next_pair = ((context >> 11) & 15) as usize;
    if previous_pair != 0 {
        push(layout::PREVIOUS_PAIR + previous_pair);
    }
    if next_pair != 0 {
        push(layout::NEXT_PAIR + next_pair);
    }
    let previous_symbol = ((context >> 22) & 31) as usize;
    let next_symbol = ((context >> 27) & 31) as usize;
    if previous_symbol != 0 {
        push(layout::PREVIOUS_SYMBOL_PAIR + previous_symbol);
    }
    if next_symbol != 0 {
        push(layout::NEXT_SYMBOL_PAIR + next_symbol);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf16(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    #[test]
    fn splits_parts_like_upstream() {
        let prepared = prepare(&utf16("let a\t= 1;\r\n// é"));
        let ranges: Vec<(u32, u32)> = prepared.ranges.chunks(2).map(|r| (r[0], r[1])).collect();
        assert_eq!(
            ranges,
            vec![(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
        );
        let kinds: Vec<u32> = (0..prepared.len()).map(|i| prepared.kind(i)).collect();
        assert_eq!(kinds, vec![0, 1, 0, 1, 3, 1, 0, 3, 2, 3, 3, 1, 0]);
    }

    #[test]
    fn layout_matches_checkpoint() {
        assert_eq!(layout::INPUT_SIZE, crate::generated::model::INPUT_SIZE);
    }

    #[test]
    fn empty_source_has_no_parts() {
        assert!(prepare(&[]).is_empty());
    }
}
