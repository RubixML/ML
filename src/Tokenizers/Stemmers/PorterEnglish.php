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
     * The word being stemmed.
     *
     * @var string
     */
    private string $word = '';

    /**
     * The offset of the last character of the word.
     *
     * @var int
     */
    private int $k = 0;

    /**
     * The offset of the first suffix character.
     *
     * @var int
     */
    private int $j = 0;

    /**
     * The offset of the first letter of the word.
     *
     * @var int
     */
    private int $k0 = 0;

    /**
     * Stem a word to its root form.
     *
     * @param string $word
     * @return string
     */
    public function stem(string $word) : string
    {
        $this->word = $word;
        $this->k0 = 0;
        $this->k = strlen($word) - 1;

        if ($this->k > $this->k0 + 1) {
            $this->stepOne();
            $this->stepTwo();
            $this->stepThree();
            $this->stepFour();
            $this->stepFive();
            $this->stepSix();
        }

        return substr($this->word, 0, $this->k + 1);
    }

    /**
     * Return true if the character at the given offset is a consonant.
     *
     * @param int $i
     * @return bool
     */
    private function cons(int $i) : bool
    {
        $ch = $this->word[$i];

        if ($ch === 'a' || $ch === 'e' || $ch === 'i' || $ch === 'o' || $ch === 'u') {
            return false;
        }

        if ($ch === 'y') {
            return $i === $this->k0 ? true : !$this->cons($i - 1);
        }

        return true;
    }

    /**
     * Measure the number of consonant sequences between the start of the
     * word and the first suffix character.
     *
     * @return int
     */
    private function measure() : int
    {
        $n = 0;
        $i = $this->k0;

        while (true) {
            if ($i > $this->j) {
                return $n;
            }

            if (!$this->cons($i)) {
                break;
            }

            ++$i;
        }

        ++$i;

        while (true) {
            while (true) {
                if ($i > $this->j) {
                    return $n;
                }

                if ($this->cons($i)) {
                    break;
                }

                ++$i;
            }

            ++$i;
            ++$n;

            while (true) {
                if ($i > $this->j) {
                    return $n;
                }

                if (!$this->cons($i)) {
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
     * @return bool
     */
    private function vowelInStem() : bool
    {
        for ($i = $this->k0; $i <= $this->j; ++$i) {
            if (!$this->cons($i)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Return true if the two characters immediately before the first suffix
     * character are identical consonants.
     *
     * @param int $j
     * @return bool
     */
    private function doubleConsonant(int $j) : bool
    {
        if ($j < $this->k0 + 1) {
            return false;
        }

        if ($this->word[$j] !== $this->word[$j - 1]) {
            return false;
        }

        return $this->cons($j);
    }

    /**
     * Return true if the last three characters before the first suffix
     * character follow a consonant-vowel-consonant pattern that does not
     * end in w, x, or y.
     *
     * @param int $i
     * @return bool
     */
    private function cvc(int $i) : bool
    {
        if ($i < $this->k0 + 2 || !$this->cons($i) || $this->cons($i - 1)
            || !$this->cons($i - 2)) {
            return false;
        }

        $ch = $this->word[$i];

        if ($ch === 'w' || $ch === 'x' || $ch === 'y') {
            return false;
        }

        return true;
    }

    /**
     * Return true if the word ends in the given suffix, marking the first
     * suffix character.
     *
     * @param string $suffix
     * @return bool
     */
    private function ends(string $suffix) : bool
    {
        $len = strlen($suffix);
        $offset = $this->k - $len + 1;

        if ($offset < $this->k0) {
            return false;
        }

        for ($i = 0; $i < $len; ++$i) {
            if ($this->word[$offset + $i] !== $suffix[$i]) {
                return false;
            }
        }

        $this->j = $this->k - $len;

        return true;
    }

    /**
     * Replace the suffix starting at the first suffix character with the
     * given string.
     *
     * @param string $suffix
     */
    private function setTo(string $suffix) : void
    {
        $len = strlen($suffix);

        for ($i = 0; $i < $len; ++$i) {
            $this->word[$this->j + 1 + $i] = $suffix[$i];
        }

        $this->k = $this->j + $len;
    }

    /**
     * Replace the suffix starting at the first suffix character with the
     * given string only if the stem has a measure greater than zero.
     *
     * @param string $suffix
     */
    private function r(string $suffix) : void
    {
        if ($this->measure() > 0) {
            $this->setTo($suffix);
        }
    }

    /**
     * Strip plurals and the endings ed, ing, and eed.
     */
    private function stepOne() : void
    {
        if ($this->word[$this->k] === 's') {
            if ($this->ends('sses')) {
                $this->k -= 2;
            } elseif ($this->ends('ies')) {
                $this->setTo('i');
            } elseif ($this->word[$this->k - 1] !== 's') {
                --$this->k;
            }
        }

        if ($this->ends('eed')) {
            if ($this->measure() > 0) {
                --$this->k;
            }
        } elseif (($this->ends('ed') || $this->ends('ing')) && $this->vowelInStem()) {
            $this->k = $this->j;

            if ($this->ends('at')) {
                $this->setTo('ate');
            } elseif ($this->ends('bl')) {
                $this->setTo('ble');
            } elseif ($this->ends('iz')) {
                $this->setTo('ize');
            } elseif ($this->doubleConsonant($this->k)) {
                $ch = $this->word[$this->k--];

                if ($ch === 'l' || $ch === 's' || $ch === 'z') {
                    ++$this->k;
                }
            } elseif ($this->measure() === 1 && $this->cvc($this->k)) {
                $this->setTo('e');
            }
        }
    }

    /**
     * Change a terminal y to i if there is another vowel in the stem.
     */
    private function stepTwo() : void
    {
        if ($this->ends('y') && $this->vowelInStem()) {
            $this->word[$this->k] = 'i';
        }
    }

    /**
     * Map double suffixes to single ones, e.g. ization to ize.
     */
    private function stepThree() : void
    {
        if ($this->k === $this->k0) {
            return;
        }

        switch ($this->word[$this->k - 1]) {
            case 'a':
                if ($this->ends('ational')) {
                    $this->r('ate');

                    break;
                }

                if ($this->ends('tional')) {
                    $this->r('tion');

                    break;
                }

                break;

            case 'c':
                if ($this->ends('enci')) {
                    $this->r('ence');

                    break;
                }

                if ($this->ends('anci')) {
                    $this->r('ance');

                    break;
                }

                break;

            case 'e':
                if ($this->ends('izer')) {
                    $this->r('ize');

                    break;
                }

                break;

            case 'l':
                if ($this->ends('bli')) {
                    $this->r('ble');

                    break;
                }

                if ($this->ends('alli')) {
                    $this->r('al');

                    break;
                }

                if ($this->ends('entli')) {
                    $this->r('ent');

                    break;
                }

                if ($this->ends('eli')) {
                    $this->r('e');

                    break;
                }

                if ($this->ends('ousli')) {
                    $this->r('ous');

                    break;
                }

                break;

            case 'o':
                if ($this->ends('ization')) {
                    $this->r('ize');

                    break;
                }

                if ($this->ends('ation')) {
                    $this->r('ate');

                    break;
                }

                if ($this->ends('ator')) {
                    $this->r('ate');

                    break;
                }

                break;

            case 's':
                if ($this->ends('alism')) {
                    $this->r('al');

                    break;
                }

                if ($this->ends('iveness')) {
                    $this->r('ive');

                    break;
                }

                if ($this->ends('fulness')) {
                    $this->r('ful');

                    break;
                }

                if ($this->ends('ousness')) {
                    $this->r('ous');

                    break;
                }

                break;

            case 't':
                if ($this->ends('aliti')) {
                    $this->r('al');

                    break;
                }

                if ($this->ends('iviti')) {
                    $this->r('ive');

                    break;
                }

                if ($this->ends('biliti')) {
                    $this->r('ble');

                    break;
                }

                break;

            case 'g':
                if ($this->ends('logi')) {
                    $this->r('log');

                    break;
                }

                break;
        }
    }

    /**
     * Remove suffixes such as ic, ful, and ness.
     */
    private function stepFour() : void
    {
        switch ($this->word[$this->k]) {
            case 'e':
                if ($this->ends('icate')) {
                    $this->r('ic');

                    break;
                }

                if ($this->ends('ative')) {
                    $this->r('');

                    break;
                }

                if ($this->ends('alize')) {
                    $this->r('al');

                    break;
                }

                break;

            case 'i':
                if ($this->ends('iciti')) {
                    $this->r('ic');

                    break;
                }

                break;

            case 'l':
                if ($this->ends('ical')) {
                    $this->r('ic');

                    break;
                }

                if ($this->ends('ful')) {
                    $this->r('');

                    break;
                }

                break;

            case 's':
                if ($this->ends('ness')) {
                    $this->r('');

                    break;
                }

                break;
        }
    }

    /**
     * Remove a suffix in the context of a measure greater than one.
     */
    private function stepFive() : void
    {
        if ($this->k === $this->k0) {
            return;
        }

        switch ($this->word[$this->k - 1]) {
            case 'a':
                if ($this->ends('al')) {
                    break;
                }

                return;

            case 'c':
                if ($this->ends('ance')) {
                    break;
                }

                if ($this->ends('ence')) {
                    break;
                }

                return;

            case 'e':
                if ($this->ends('er')) {
                    break;
                }

                return;

            case 'i':
                if ($this->ends('ic')) {
                    break;
                }

                return;

            case 'l':
                if ($this->ends('able')) {
                    break;
                }

                if ($this->ends('ible')) {
                    break;
                }

                return;

            case 'n':
                if ($this->ends('ant')) {
                    break;
                }

                if ($this->ends('ement')) {
                    break;
                }

                if ($this->ends('ment')) {
                    break;
                }

                if ($this->ends('ent')) {
                    break;
                }

                return;

            case 'o':
                if ($this->ends('ion') && $this->j >= 0
                    && ($this->word[$this->j] === 's' || $this->word[$this->j] === 't')) {
                    break;
                }

                if ($this->ends('ou')) {
                    break;
                }

                return;

            case 's':
                if ($this->ends('ism')) {
                    break;
                }

                return;

            case 't':
                if ($this->ends('ate')) {
                    break;
                }

                if ($this->ends('iti')) {
                    break;
                }

                return;

            case 'u':
                if ($this->ends('ous')) {
                    break;
                }

                return;

            case 'v':
                if ($this->ends('ive')) {
                    break;
                }

                return;

            case 'z':
                if ($this->ends('ize')) {
                    break;
                }

                return;

            default:
                return;
        }

        if ($this->measure() > 1) {
            $this->k = $this->j;
        }
    }

    /**
     * Remove a final e if the measure allows it and undouble a final l.
     */
    private function stepSix() : void
    {
        $this->j = $this->k;

        if ($this->word[$this->k] === 'e') {
            $a = $this->measure();

            if ($a > 1 || ($a === 1 && !$this->cvc($this->k - 1))) {
                --$this->k;
            }
        }

        if ($this->word[$this->k] === 'l' && $this->doubleConsonant($this->k)
            && $this->measure() > 1) {
            --$this->k;
        }
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
