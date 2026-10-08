<?php

namespace Rubix\ML\Tokenizers\Stemmers;

/**
 * Porter English
 *
 * A pure PHP implementation of the Porter stemming algorithm for English.
 *
 * References:
 * [1] M. F. Porter. (1980). An algorithm for suffix stripping. Program, 14(3), 130-137.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PorterEnglish implements Stemmer
{
    /**
     * Return true if the character at the given offset is a consonant.
     *
     * @param string $word
     * @param int $offset
     * @return bool
     */
    protected static function cons(string $word, int $offset) : bool
    {
        $character = $word[$offset];

        if ($character === 'a' or $character === 'e' or $character === 'i' or $character === 'o' or $character === 'u') {
            return false;
        }

        if ($character === 'y') {
            return $offset === 0 ? true : !self::cons($word, $offset - 1);
        }

        return true;
    }

    /**
     * Measure the number of consonant sequences between the start of the
     * word and the end of the stem.
     *
     * @param string $word
     * @param int $stemEnd
     * @return int
     */
    protected static function measure(string $word, int $stemEnd) : int
    {
        $sequences = 0;
        $offset = 0;

        while (true) {
            if ($offset > $stemEnd) {
                return $sequences;
            }

            if (!self::cons($word, $offset)) {
                break;
            }

            ++$offset;
        }

        ++$offset;

        while (true) {
            while (true) {
                if ($offset > $stemEnd) {
                    return $sequences;
                }

                if (self::cons($word, $offset)) {
                    break;
                }

                ++$offset;
            }

            ++$offset;
            ++$sequences;

            while (true) {
                if ($offset > $stemEnd) {
                    return $sequences;
                }

                if (!self::cons($word, $offset)) {
                    break;
                }

                ++$offset;
            }

            ++$offset;
        }
    }

    /**
     * Return true if the region from the start of the word to the end of the
     * stem contains a vowel.
     *
     * @param string $word
     * @param int $stemEnd
     * @return bool
     */
    protected static function vowelInStem(string $word, int $stemEnd) : bool
    {
        for ($offset = 0; $offset <= $stemEnd; ++$offset) {
            if (!self::cons($word, $offset)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Return true if the character at the given offset and the one
     * before it are identical consonants.
     *
     * @param string $word
     * @param int $offset
     * @return bool
     */
    protected static function doubleConsonant(string $word, int $offset) : bool
    {
        if ($offset < 1) {
            return false;
        }

        if ($word[$offset] !== $word[$offset - 1]) {
            return false;
        }

        return self::cons($word, $offset);
    }

    /**
     * Return true if the last three characters before the given offset follow a
     * consonant-vowel-consonant pattern that does not end in w, x, or y.
     *
     * @param string $word
     * @param int $offset
     * @return bool
     */
    protected static function cvc(string $word, int $offset) : bool
    {
        if ($offset < 2 or !self::cons($word, $offset) or self::cons($word, $offset - 1) or !self::cons($word, $offset - 2)) {
            return false;
        }

        $character = $word[$offset];

        if ($character === 'w' or $character === 'x' or $character === 'y') {
            return false;
        }

        return true;
    }

    /**
     * Return true if the word ends in the given suffix, marking the end of
     * the stem.
     *
     * @param string $word
     * @param int $wordEnd
     * @param string $suffix
     * @param int $stemEnd
     * @return bool
     */
    protected static function ends(string $word, int $wordEnd, string $suffix, int &$stemEnd) : bool
    {
        $length = strlen($suffix);

        $start = $wordEnd - $length + 1;

        if ($start < 0) {
            return false;
        }

        for ($offset = 0; $offset < $length; ++$offset) {
            if ($word[$start + $offset] !== $suffix[$offset]) {
                return false;
            }
        }

        $stemEnd = $wordEnd - $length;

        return true;
    }

    /**
     * Replace the suffix that follows the end of the stem with the given
     * string, extending or truncating the word as needed.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     * @param string $suffix
     */
    protected static function setTo(string &$word, int &$wordEnd, int $stemEnd, string $suffix) : void
    {
        $length = strlen($suffix);

        for ($offset = 0; $offset < $length; ++$offset) {
            $word[$stemEnd + 1 + $offset] = $suffix[$offset];
        }

        $wordEnd = $stemEnd + $length;
    }

    /**
     * Replace the suffix that follows the end of the stem with the given
     * string only if the stem is measurable, that is, its measure is
     * greater than zero.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     * @param string $suffix
     */
    protected static function replaceIfMeasurable(string &$word, int &$wordEnd, int $stemEnd, string $suffix) : void
    {
        if (self::measure($word, $stemEnd) > 0) {
            self::setTo($word, $wordEnd, $stemEnd, $suffix);
        }
    }

    /**
     * Strip plurals and the endings ed, ing, and eed.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     */
    protected static function stepOne(string &$word, int &$wordEnd, int &$stemEnd) : void
    {
        if ($word[$wordEnd] === 's') {
            if (self::ends($word, $wordEnd, 'sses', $stemEnd)) {
                $wordEnd -= 2;
            } elseif (self::ends($word, $wordEnd, 'ies', $stemEnd)) {
                self::setTo($word, $wordEnd, $stemEnd, 'i');
            } elseif ($word[$wordEnd - 1] !== 's') {
                --$wordEnd;
            }
        }

        if (self::ends($word, $wordEnd, 'eed', $stemEnd)) {
            if (self::measure($word, $stemEnd) > 0) {
                --$wordEnd;
            }
        } elseif ((self::ends($word, $wordEnd, 'ed', $stemEnd) or self::ends($word, $wordEnd, 'ing', $stemEnd)) and self::vowelInStem($word, $stemEnd)) {
            $wordEnd = $stemEnd;

            if (self::ends($word, $wordEnd, 'at', $stemEnd)) {
                self::setTo($word, $wordEnd, $stemEnd, 'ate');
            } elseif (self::ends($word, $wordEnd, 'bl', $stemEnd)) {
                self::setTo($word, $wordEnd, $stemEnd, 'ble');
            } elseif (self::ends($word, $wordEnd, 'iz', $stemEnd)) {
                self::setTo($word, $wordEnd, $stemEnd, 'ize');
            } elseif (self::doubleConsonant($word, $wordEnd)) {
                $character = $word[$wordEnd];

                --$wordEnd;

                if ($character === 'l' or $character === 's' or $character === 'z') {
                    ++$wordEnd;
                }
            } elseif (self::measure($word, $stemEnd) === 1 and self::cvc($word, $wordEnd)) {
                self::setTo($word, $wordEnd, $stemEnd, 'e');
            }
        }
    }

    /**
     * Change a terminal y to i if there is another vowel in the stem.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     */
    protected static function stepTwo(string &$word, int $wordEnd, int &$stemEnd) : void
    {
        if (self::ends($word, $wordEnd, 'y', $stemEnd) and self::vowelInStem($word, $stemEnd)) {
            $word[$wordEnd] = 'i';
        }
    }

    /**
     * Map double suffixes to single ones, e.g. ization to ize.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     */
    protected static function stepThree(string &$word, int &$wordEnd, int &$stemEnd) : void
    {
        if ($wordEnd === 0) {
            return;
        }

        switch ($word[$wordEnd - 1]) {
            case 'a':
                if (self::ends($word, $wordEnd, 'ational', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ate');

                    break;
                }

                if (self::ends($word, $wordEnd, 'tional', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'tion');

                    break;
                }

                break;

            case 'c':
                if (self::ends($word, $wordEnd, 'enci', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ence');

                    break;
                }

                if (self::ends($word, $wordEnd, 'anci', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ance');

                    break;
                }

                break;

            case 'e':
                if (self::ends($word, $wordEnd, 'izer', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ize');

                    break;
                }

                break;

            case 'l':
                if (self::ends($word, $wordEnd, 'bli', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ble');

                    break;
                }

                if (self::ends($word, $wordEnd, 'alli', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'al');

                    break;
                }

                if (self::ends($word, $wordEnd, 'entli', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ent');

                    break;
                }

                if (self::ends($word, $wordEnd, 'eli', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'e');

                    break;
                }

                if (self::ends($word, $wordEnd, 'ousli', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ous');

                    break;
                }

                break;

            case 'o':
                if (self::ends($word, $wordEnd, 'ization', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ize');

                    break;
                }

                if (self::ends($word, $wordEnd, 'ation', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ate');

                    break;
                }

                if (self::ends($word, $wordEnd, 'ator', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ate');

                    break;
                }

                break;

            case 's':
                if (self::ends($word, $wordEnd, 'alism', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'al');

                    break;
                }

                if (self::ends($word, $wordEnd, 'iveness', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ive');

                    break;
                }

                if (self::ends($word, $wordEnd, 'fulness', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ful');

                    break;
                }

                if (self::ends($word, $wordEnd, 'ousness', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ous');

                    break;
                }

                break;

            case 't':
                if (self::ends($word, $wordEnd, 'aliti', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'al');

                    break;
                }

                if (self::ends($word, $wordEnd, 'iviti', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ive');

                    break;
                }

                if (self::ends($word, $wordEnd, 'biliti', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ble');

                    break;
                }

                break;

            case 'g':
                if (self::ends($word, $wordEnd, 'logi', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'log');

                    break;
                }

                break;
        }
    }

    /**
     * Remove suffixes such as ic, ful, and ness.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     */
    protected static function stepFour(string &$word, int &$wordEnd, int &$stemEnd) : void
    {
        switch ($word[$wordEnd]) {
            case 'e':
                if (self::ends($word, $wordEnd, 'icate', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ic');

                    break;
                }

                if (self::ends($word, $wordEnd, 'ative', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, '');

                    break;
                }

                if (self::ends($word, $wordEnd, 'alize', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'al');

                    break;
                }

                break;

            case 'i':
                if (self::ends($word, $wordEnd, 'iciti', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ic');

                    break;
                }

                break;

            case 'l':
                if (self::ends($word, $wordEnd, 'ical', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, 'ic');

                    break;
                }

                if (self::ends($word, $wordEnd, 'ful', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, '');

                    break;
                }

                break;

            case 's':
                if (self::ends($word, $wordEnd, 'ness', $stemEnd)) {
                    self::replaceIfMeasurable($word, $wordEnd, $stemEnd, '');

                    break;
                }

                break;
        }
    }

    /**
     * Remove a suffix in the context of a measure greater than one.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     */
    protected static function stepFive(string &$word, int &$wordEnd, int &$stemEnd) : void
    {
        if ($wordEnd === 0) {
            return;
        }

        switch ($word[$wordEnd - 1]) {
            case 'a':
                if (self::ends($word, $wordEnd, 'al', $stemEnd)) {
                    break;
                }

                return;

            case 'c':
                if (self::ends($word, $wordEnd, 'ance', $stemEnd)) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'ence', $stemEnd)) {
                    break;
                }

                return;

            case 'e':
                if (self::ends($word, $wordEnd, 'er', $stemEnd)) {
                    break;
                }

                return;

            case 'i':
                if (self::ends($word, $wordEnd, 'ic', $stemEnd)) {
                    break;
                }

                return;

            case 'l':
                if (self::ends($word, $wordEnd, 'able', $stemEnd)) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'ible', $stemEnd)) {
                    break;
                }

                return;

            case 'n':
                if (self::ends($word, $wordEnd, 'ant', $stemEnd)) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'ement', $stemEnd)) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'ment', $stemEnd)) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'ent', $stemEnd)) {
                    break;
                }

                return;

            case 'o':
                if (self::ends($word, $wordEnd, 'ion', $stemEnd) and $stemEnd >= 0 and ($word[$stemEnd] === 's' or $word[$stemEnd] === 't')) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'ou', $stemEnd)) {
                    break;
                }

                return;

            case 's':
                if (self::ends($word, $wordEnd, 'ism', $stemEnd)) {
                    break;
                }

                return;

            case 't':
                if (self::ends($word, $wordEnd, 'ate', $stemEnd)) {
                    break;
                }

                if (self::ends($word, $wordEnd, 'iti', $stemEnd)) {
                    break;
                }

                return;

            case 'u':
                if (self::ends($word, $wordEnd, 'ous', $stemEnd)) {
                    break;
                }

                return;

            case 'v':
                if (self::ends($word, $wordEnd, 'ive', $stemEnd)) {
                    break;
                }

                return;

            case 'z':
                if (self::ends($word, $wordEnd, 'ize', $stemEnd)) {
                    break;
                }

                return;

            default:
                return;
        }

        if (self::measure($word, $stemEnd) > 1) {
            $wordEnd = $stemEnd;
        }
    }

    /**
     * Remove a final e if the measure allows it and undouble a final l.
     *
     * @param string $word
     * @param int $wordEnd
     * @param int $stemEnd
     */
    protected static function stepSix(string &$word, int &$wordEnd, int &$stemEnd) : void
    {
        $stemEnd = $wordEnd;

        if ($word[$wordEnd] === 'e') {
            $stemMeasure = self::measure($word, $stemEnd);

            if ($stemMeasure > 1 or ($stemMeasure === 1 && !self::cvc($word, $wordEnd - 1))) {
                --$wordEnd;
            }
        }

        if ($word[$wordEnd] === 'l' and self::doubleConsonant($word, $wordEnd) and self::measure($word, $stemEnd) > 1) {
            --$wordEnd;
        }
    }

    /**
     * Stem a word to its root form.
     *
     * @param string $word
     * @return string
     */
    public function stem(string $word) : string
    {
        $wordEnd = strlen($word) - 1;

        if ($wordEnd > 1) {
            $stemEnd = 0;

            self::stepOne($word, $wordEnd, $stemEnd);
            self::stepTwo($word, $wordEnd, $stemEnd);
            self::stepThree($word, $wordEnd, $stemEnd);
            self::stepFour($word, $wordEnd, $stemEnd);
            self::stepFive($word, $wordEnd, $stemEnd);
            self::stepSix($word, $wordEnd, $stemEnd);
        }

        return substr($word, 0, $wordEnd + 1);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Porter English';
    }
}
