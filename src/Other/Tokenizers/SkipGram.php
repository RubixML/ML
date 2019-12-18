<?php

namespace Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;

use function count;

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
 */
class SkipGram implements Tokenizer
{
    protected const SENTENCE_REGEX = '/(?<=[.?!])\s+(?=[a-z])/i';
    protected const WORD_REGEX = '/\w+/u';

    protected const SEPARATOR = ' ';

    /**
     * The number of contiguous words to a single token.
     *
     * @var int
     */
    protected $n;

    /**
     * The number of words to skip over to form new sequences.
     *
     * @var int
     */
    protected $skip;

    /**
     * @param int $n
     * @param int $skip
     * @throws \InvalidArgumentException
     */
    public function __construct(int $n = 2, int $skip = 2)
    {
        if ($n < 2) {
            throw new InvalidArgumentException('The number of words'
                . " per token must be greater than 1, $n given.");
        }

        if ($skip < 2) {
            throw new InvalidArgumentException('The number of words'
                . " to skip must be greater than 1, $skip given.");
        }

        $this->n = $n;
        $this->skip = $skip;
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return string[]
     */
    public function tokenize(string $string) : array
    {
        $sentences = preg_split(self::SENTENCE_REGEX, $string) ?: [];

        $tokens = [];

        foreach ($sentences as $sentence) {
            $words = [];

            preg_match_all(self::WORD_REGEX, $sentence, $words);

            $words = $words[0];

            $length = count($words) - ($this->n + $this->skip);

            for ($i = 0; $i <= $length; ++$i) {
                $first = $words[$i];

                for ($j = 0; $j <= $this->skip; ++$j) {
                    $skipGram = $first;

                    for ($k = 1; $k < $this->n; ++$k) {
                        $skipGram .= self::SEPARATOR . $words[$i + $j + $k];
                    }

                    $tokens[] = $skipGram;
                }
            }

            $last = array_slice($words, -$this->n);

            $tokens[] = implode(self::SEPARATOR, $last);
        }

        return $tokens;
    }
}
