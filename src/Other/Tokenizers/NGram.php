<?php

namespace Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;

/**
 * N-Gram
 * 
 * N-Grams are sequences of n-words of a given string. For example, if *n*
 * is 2 then the tokenizer will generate tokens consisting of 2 contiguous
 * words.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NGram implements Tokenizer
{
    const SENTENCE_REGEX = '/(?<=[.?!])\s+(?=[a-z])/i';
    const WORD_REGEX = '/\w+/u';

    const SEPARATOR = ' ';

    /**
     * The number of contiguous words to a single token.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $n = 2)
    {
        if ($n < 2) {
            throw new InvalidArgumentException('The number of words'
                . " per token must be greater than 1, $n given.");
        }

        $this->n = $n;
    }

    /**
     * Tokenize a block of text.
     *
     * @param  string  $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        $sentences = preg_split(self::SENTENCE_REGEX, $string) ?: [];

        $nGrams = [];

        foreach ($sentences as $sentence) {
            $words = [];

            preg_match_all(self::WORD_REGEX, $sentence, $words);

            $words = $words[0];

            $length = count($words) - $this->n + 1;

            for ($i = 0; $i < $length; $i++) {
                $nGram = '';

                for ($j = 0; $j < $this->n; $j++) {
                    $nGram .= $words[$i + $j];

                    if ($j < $this->n - 1) {
                        $nGram .= self::SEPARATOR;
                    }
                }

                $nGrams[] = $nGram;
            }
        }

        return $nGrams;
    }
}
