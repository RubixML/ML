<?php

namespace Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;

/**
 * N-gram
 *
 * N-grams are sequences of n-words of a given string. The N-gram tokenizer
 * outputs tokens of contiguous words ranging from *min* to *max* number of
 * words per token.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NGram implements Tokenizer
{
    protected const WORD_REGEX = '/\w+/u';

    protected const SENTENCE_REGEX = '/(?<=[.?!])\s+(?=[a-z])/i';

    protected const SEPARATOR = ' ';

    /**
     * The minimum number of contiguous words in a single token.
     *
     * @var int
     */
    protected $min;

    /**
     * The maximum number of contiguous words in a single token.
     *
     * @var int
     */
    protected $max;

    /**
     * @param int $min
     * @param int $max
     * @throws \InvalidArgumentException
     */
    public function __construct(int $min = 2, int $max = 2)
    {
        if ($min < 1) {
            throw new InvalidArgumentException('Minimum cannot be less'
                . ' than minimum.');
        }

        if ($min > $max) {
            throw new InvalidArgumentException('Minimum cannot be greater'
                . ' than maximum.');
        }

        $this->min = $min;
        $this->max = $max;
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        $sentences = preg_split(self::SENTENCE_REGEX, $string) ?: [];

        $tokens = [];

        foreach ($sentences as $sentence) {
            $words = [];

            preg_match_all(self::WORD_REGEX, $sentence, $words);

            $words = $words[0];

            $n = count($words);

            foreach ($words as $i => $word) {
                $p = min($n - $i, $this->max);

                for ($j = $this->min; $j <= $p; ++$j) {
                    $nGram = $word;

                    for ($k = 1; $k < $j; ++$k) {
                        $nGram .= self::SEPARATOR . $words[$i + $k];
                    }

                    $tokens[] = $nGram;
                }
            }
        }

        return $tokens;
    }
}
