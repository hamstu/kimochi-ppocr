//! Corrections for Japanese text read by PP-OCR's multilingual (largely Chinese-trained) recognizer.
//!
//! Two slips recur on game text: a Chinese form where Japanese uses a different glyph (`值` for
//! `値`), and hiragana へ/べ/ぺ inside a katakana word, which look identical. Both are fixed in place
//! on the line text and its characters; boxes and confidences are unchanged.
use crate::TextLine;

/// Chinese forms the recognizer emits for Japanese glyphs. Only characters that are not used in
/// Japanese text themselves belong here.
const VARIANTS: &[(char, char)] = &[('值', '値'), ('说', '説'), ('别', '別'), ('练', '練'), ('级', '級')];

fn katakana(c: char) -> bool {
    matches!(c, '\u{30A1}'..='\u{30FA}' | 'ー' | '・')
}

fn to_katakana(c: char) -> Option<char> {
    match c {
        'へ' => Some('ヘ'),
        'べ' => Some('ベ'),
        'ぺ' => Some('ペ'),
        _ => None,
    }
}

pub fn fix_lines(lines: &mut [TextLine]) {
    for line in lines {
        let chars: Vec<char> = line.characters.iter().map(|c| c.character).collect();
        for (i, character) in line.characters.iter_mut().enumerate() {
            let c = chars[i];
            if let Some(&(_, ja)) = VARIANTS.iter().find(|(cn, _)| *cn == c) {
                character.character = ja;
            } else if let Some(k) = to_katakana(c) {
                let before = i > 0 && katakana(chars[i - 1]);
                let after = chars.get(i + 1).is_some_and(|&n| katakana(n));
                if before || after {
                    character.character = k;
                }
            }
        }
        line.text = line.characters.iter().map(|c| c.character).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Character;
    fn line(text: &str) -> TextLine {
        TextLine {
            text: text.into(),
            bbox: [0, 0, 1, 1],
            detection_confidence: None,
            characters: text
                .chars()
                .map(|character| Character { character, bbox: [0, 0, 1, 1], confidence: 1.0 })
                .collect(),
            is_vertical: false,
        }
    }
    #[test]
    fn fixes_katakana_context_and_chinese_variants_only() {
        let mut lines = [line("レべルが上がった"), line("経験值"), line("図書館へ行く")];
        fix_lines(&mut lines);
        assert_eq!(lines[0].text, "レベルが上がった");
        assert_eq!(lines[1].text, "経験値");
        assert_eq!(lines[2].text, "図書館へ行く");
    }
}
