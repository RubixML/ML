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
     * @param int $i
     * @return bool
     */
    protected static function cons(string $word, int $i) : bool
    {
        $ch = $word[$i];

        if ($ch === 'a' || $ch === 'e' || $ch === 'i' || $ch === 'o' || $ch === 'u') {
            return false;
        }

        if ($ch === 'y') {
            return $i === 0 ? true : !self::cons($word, $i - 1);
        }

        return true;
    }

    /**
     * Measure the number of consonant sequences between the start of the
     * word and the first suffix character.
     *
     * @param string $word
     * @param int $j
     * @return int
     */
    protected static function measure(string $word, int $j) : int
    {
        $n = 0;
        $i = 0;

        while (true) {
            if ($i > $j) {
                return $n;
            }

            if (!self::cons($word, $i)) {
                break;
            }

            ++$i;
        }

        ++$i;

        while (true) {
            while (true) {
                if ($i > $j) {
                    return $n;
                }

                if (self::cons($word, $i)) {
                    break;
                }

                ++$i;
            }

            ++$i;
            ++$n;

            while (true) {
                if ($i > $j) {
                    return $n;
                }

                if (!self::cons($word, $i)) {
                    break;
                }

                ++$i;
            }

            ++$i;
        }
    }

    /**
     * Return true if the region from the start of the word to the first
     * suffix character contains a vowel.
     *
     * @param string $word
     * @param int $j
     * @return bool
     */
    protected static function vowelInStem(string $word, int $j) : bool
    {
        for ($i = 0; $i <= $j; ++$i) {
            if (!self::cons($word, $i)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Return true if the two characters immediately before the first suffix
     * character are identical consonants.
     *
     * @param string $word
     * @param int $i
     * @return bool
     */
    protected static function doubleConsonant(string $word, int $i) : bool
    {
        if ($i < 1) {
            return false;
        }

        if ($word[$i] !== $word[$i - 1]) {
            return false;
        }

        return self::cons($word, $i);
    }

    /**
     * Return true if the last three characters before the first suffix
     * character follow a consonant-vowel-consonant pattern that does not
     * end in w, x, or y.
     *
     * @param string $word
     * @param int $i
     * @return bool
     */
    protected static function cvc(string $word, int $i) : bool
    {
        if ($i < 2 || !self::cons($word, $i) || self::cons($word, $i - 1)
            || !self::cons($word, $i - 2)) {
            return false;
        }

        $ch = $word[$i];

        if ($ch === 'w' || $ch === 'x' || $ch === 'y') {
            return false;
        }

        return true;
    }

    /**
     * Return true if the word ends in the given suffix, marking the first
     * suffix character.
     *
     * @param string $word
     * @param int $k
     * @param string $suffix
     * @param int $j
     * @return bool
     */
    protected static function ends(string $word, int $k, string $suffix, int &$j) : bool
    {
        $len = strlen($suffix);
        $offset = $k - $len + 1;

        if ($offset < 0) {
            return false;
        }

        for ($i = 0; $i < $len; ++$i) {
            if ($word[$offset + $i] !== $suffix[$i]) {
                return false;
            }
        }

        $j = $k - $len;

        return true;
    }

    /**
     * Replace the suffix starting at the first suffix character with the
     * given string.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     * @param string $suffix
     */
    protected static function setTo(string &$word, int &$k, int $j, string $suffix) : void
    {
        $len = strlen($suffix);

        for ($i = 0; $i < $len; ++$i) {
            $word[$j + 1 + $i] = $suffix[$i];
        }

        $k = $j + $len;
    }

    /**
     * Replace the suffix starting at the first suffix character with the
     * given string only if the stem has a measure greater than zero.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     * @param string $suffix
     */
    protected static function r(string &$word, int &$k, int $j, string $suffix) : void
    {
        if (self::measure($word, $j) > 0) {
            self::setTo($word, $k, $j, $suffix);
        }
    }

    /**
     * Strip plurals and the endings ed, ing, and eed.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     */
    protected static function stepOne(string &$word, int &$k, int &$j) : void
    {
        if ($word[$k] === 's') {
            if (self::ends($word, $k, 'sses', $j)) {
                $k -= 2;
            } elseif (self::ends($word, $k, 'ies', $j)) {
                self::setTo($word, $k, $j, 'i');
            } elseif ($word[$k - 1] !== 's') {
                --$k;
            }
        }

        if (self::ends($word, $k, 'eed', $j)) {
            if (self::measure($word, $j) > 0) {
                --$k;
            }
        } elseif ((self::ends($word, $k, 'ed', $j) || self::ends($word, $k, 'ing', $j))
            && self::vowelInStem($word, $j)) {
            $k = $j;

            if (self::ends($word, $k, 'at', $j)) {
                self::setTo($word, $k, $j, 'ate');
            } elseif (self::ends($word, $k, 'bl', $j)) {
                self::setTo($word, $k, $j, 'ble');
            } elseif (self::ends($word, $k, 'iz', $j)) {
                self::setTo($word, $k, $j, 'ize');
            } elseif (self::doubleConsonant($word, $k)) {
                $ch = $word[$k--];

                if ($ch === 'l' || $ch === 's' || $ch === 'z') {
                    ++$k;
                }
            } elseif (self::measure($word, $j) === 1 && self::cvc($word, $k)) {
                self::setTo($word, $k, $j, 'e');
            }
        }
    }

    /**
     * Change a terminal y to i if there is another vowel in the stem.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     */
    protected static function stepTwo(string &$word, int $k, int &$j) : void
    {
        if (self::ends($word, $k, 'y', $j) && self::vowelInStem($word, $j)) {
            $word[$k] = 'i';
        }
    }

    /**
     * Map double suffixes to single ones, e.g. ization to ize.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     */
    protected static function stepThree(string &$word, int &$k, int &$j) : void
    {
        if ($k === 0) {
            return;
        }

        switch ($word[$k - 1]) {
            case 'a':
                if (self::ends($word, $k, 'ational', $j)) {
                    self::r($word, $k, $j, 'ate');

                    break;
                }

                if (self::ends($word, $k, 'tional', $j)) {
                    self::r($word, $k, $j, 'tion');

                    break;
                }

                break;

            case 'c':
                if (self::ends($word, $k, 'enci', $j)) {
                    self::r($word, $k, $j, 'ence');

                    break;
                }

                if (self::ends($word, $k, 'anci', $j)) {
                    self::r($word, $k, $j, 'ance');

                    break;
                }

                break;

            case 'e':
                if (self::ends($word, $k, 'izer', $j)) {
                    self::r($word, $k, $j, 'ize');

                    break;
                }

                break;

            case 'l':
                if (self::ends($word, $k, 'bli', $j)) {
                    self::r($word, $k, $j, 'ble');

                    break;
                }

                if (self::ends($word, $k, 'alli', $j)) {
                    self::r($word, $k, $j, 'al');

                    break;
                }

                if (self::ends($word, $k, 'entli', $j)) {
                    self::r($word, $k, $j, 'ent');

                    break;
                }

                if (self::ends($word, $k, 'eli', $j)) {
                    self::r($word, $k, $j, 'e');

                    break;
                }

                if (self::ends($word, $k, 'ousli', $j)) {
                    self::r($word, $k, $j, 'ous');

                    break;
                }

                break;

            case 'o':
                if (self::ends($word, $k, 'ization', $j)) {
                    self::r($word, $k, $j, 'ize');

                    break;
                }

                if (self::ends($word, $k, 'ation', $j)) {
                    self::r($word, $k, $j, 'ate');

                    break;
                }

                if (self::ends($word, $k, 'ator', $j)) {
                    self::r($word, $k, $j, 'ate');

                    break;
                }

                break;

            case 's':
                if (self::ends($word, $k, 'alism', $j)) {
                    self::r($word, $k, $j, 'al');

                    break;
                }

                if (self::ends($word, $k, 'iveness', $j)) {
                    self::r($word, $k, $j, 'ive');

                    break;
                }

                if (self::ends($word, $k, 'fulness', $j)) {
                    self::r($word, $k, $j, 'ful');

                    break;
                }

                if (self::ends($word, $k, 'ousness', $j)) {
                    self::r($word, $k, $j, 'ous');

                    break;
                }

                break;

            case 't':
                if (self::ends($word, $k, 'aliti', $j)) {
                    self::r($word, $k, $j, 'al');

                    break;
                }

                if (self::ends($word, $k, 'iviti', $j)) {
                    self::r($word, $k, $j, 'ive');

                    break;
                }

                if (self::ends($word, $k, 'biliti', $j)) {
                    self::r($word, $k, $j, 'ble');

                    break;
                }

                break;

            case 'g':
                if (self::ends($word, $k, 'logi', $j)) {
                    self::r($word, $k, $j, 'log');

                    break;
                }

                break;
        }
    }

    /**
     * Remove suffixes such as ic, ful, and ness.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     */
    protected static function stepFour(string &$word, int &$k, int &$j) : void
    {
        switch ($word[$k]) {
            case 'e':
                if (self::ends($word, $k, 'icate', $j)) {
                    self::r($word, $k, $j, 'ic');

                    break;
                }

                if (self::ends($word, $k, 'ative', $j)) {
                    self::r($word, $k, $j, '');

                    break;
                }

                if (self::ends($word, $k, 'alize', $j)) {
                    self::r($word, $k, $j, 'al');

                    break;
                }

                break;

            case 'i':
                if (self::ends($word, $k, 'iciti', $j)) {
                    self::r($word, $k, $j, 'ic');

                    break;
                }

                break;

            case 'l':
                if (self::ends($word, $k, 'ical', $j)) {
                    self::r($word, $k, $j, 'ic');

                    break;
                }

                if (self::ends($word, $k, 'ful', $j)) {
                    self::r($word, $k, $j, '');

                    break;
                }

                break;

            case 's':
                if (self::ends($word, $k, 'ness', $j)) {
                    self::r($word, $k, $j, '');

                    break;
                }

                break;
        }
    }

    /**
     * Remove a suffix in the context of a measure greater than one.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     */
    protected static function stepFive(string &$word, int &$k, int &$j) : void
    {
        if ($k === 0) {
            return;
        }

        switch ($word[$k - 1]) {
            case 'a':
                if (self::ends($word, $k, 'al', $j)) {
                    break;
                }

                return;

            case 'c':
                if (self::ends($word, $k, 'ance', $j)) {
                    break;
                }

                if (self::ends($word, $k, 'ence', $j)) {
                    break;
                }

                return;

            case 'e':
                if (self::ends($word, $k, 'er', $j)) {
                    break;
                }

                return;

            case 'i':
                if (self::ends($word, $k, 'ic', $j)) {
                    break;
                }

                return;

            case 'l':
                if (self::ends($word, $k, 'able', $j)) {
                    break;
                }

                if (self::ends($word, $k, 'ible', $j)) {
                    break;
                }

                return;

            case 'n':
                if (self::ends($word, $k, 'ant', $j)) {
                    break;
                }

                if (self::ends($word, $k, 'ement', $j)) {
                    break;
                }

                if (self::ends($word, $k, 'ment', $j)) {
                    break;
                }

                if (self::ends($word, $k, 'ent', $j)) {
                    break;
                }

                return;

            case 'o':
                if (self::ends($word, $k, 'ion', $j) && $j >= 0
                    && ($word[$j] === 's' || $word[$j] === 't')) {
                    break;
                }

                if (self::ends($word, $k, 'ou', $j)) {
                    break;
                }

                return;

            case 's':
                if (self::ends($word, $k, 'ism', $j)) {
                    break;
                }

                return;

            case 't':
                if (self::ends($word, $k, 'ate', $j)) {
                    break;
                }

                if (self::ends($word, $k, 'iti', $j)) {
                    break;
                }

                return;

            case 'u':
                if (self::ends($word, $k, 'ous', $j)) {
                    break;
                }

                return;

            case 'v':
                if (self::ends($word, $k, 'ive', $j)) {
                    break;
                }

                return;

            case 'z':
                if (self::ends($word, $k, 'ize', $j)) {
                    break;
                }

                return;

            default:
                return;
        }

        if (self::measure($word, $j) > 1) {
            $k = $j;
        }
    }

    /**
     * Remove a final e if the measure allows it and undouble a final l.
     *
     * @param string $word
     * @param int $k
     * @param int $j
     */
    protected static function stepSix(string &$word, int &$k, int &$j) : void
    {
        $j = $k;

        if ($word[$k] === 'e') {
            $a = self::measure($word, $j);

            if ($a > 1 || ($a === 1 && !self::cvc($word, $k - 1))) {
                --$k;
            }
        }

        if ($word[$k] === 'l' && self::doubleConsonant($word, $k)
            && self::measure($word, $j) > 1) {
            --$k;
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
        $k = strlen($word) - 1;

        if ($k > 1) {
            $j = 0;

            self::stepOne($word, $k, $j);
            self::stepTwo($word, $k, $j);
            self::stepThree($word, $k, $j);
            self::stepFour($word, $k, $j);
            self::stepFive($word, $k, $j);
            self::stepSix($word, $k, $j);
        }

        return substr($word, 0, $k + 1);
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
