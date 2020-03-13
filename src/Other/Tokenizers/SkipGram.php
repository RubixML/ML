<?php

namespace Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;

use function count;
use function array_slice;

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
    /**
     * The regular expression to match sentences in a blob of text.
     *
     * @var string
     */
    protected const SENTENCE_REGEX = '/(?<=[.?!])\s+(?=[a-z])/i';

    /**
     * The separator between words in the n-gram.
     *
     * @var string
     */
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
     * The word tokenizer.
     *
     * @var \Rubix\ML\Other\Tokenizers\Word
     */
    protected $wordTokenizer;

    /**
     * @param int $n
     * @param int $skip
     * @param \Rubix\ML\Other\Tokenizers\Word|null $wordTokenizer
     * @throws \InvalidArgumentException
     */
    public function __construct(int $n = 2, int $skip = 2, Word $wordTokenizer = null)
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
        $this->wordTokenizer = $wordTokenizer ?? new Word();
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
            $words = $this->wordTokenizer->tokenize($sentence);

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
