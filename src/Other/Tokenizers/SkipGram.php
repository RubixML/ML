<?php

namespace Rubix\ML\Other\Tokenizers;

use function Rubix\ML\warn_deprecated;

/**
 * Skip Gram
 *
 * Skip-grams are a technique similar to n-grams, whereby n-grams are formed but
 * in addition to allowing adjacent sequences of words, the next *k* words will
 * be skipped forming n-grams of the new forward looking sequences.
 *
 * References:
 * [1] D. Guthrie et al. A Closer Look at Skip-gram Modelling.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @deprecated
 */
class SkipGram extends KSkipNGram
{
    /**
     * @param int $n
     * @param int $skip
     * @param \Rubix\ML\Other\Tokenizers\Word|null $wordTokenizer
     */
    public function __construct(int $n = 2, int $skip = 2, ?Word $wordTokenizer = null)
    {
        warn_deprecated('Skip Gram is deprecated. Use K-Skip-N-Gram with min and max set to n instead.');

        parent::__construct($n, $n, $skip, $wordTokenizer);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Skip Gram (n: {$this->min}, skip: {$this->skip}, word tokenizer: {$this->wordTokenizer})";
    }
}
