<?php

namespace Rubix\ML\Other\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

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
class SkipGram implements Tokenizer, Stringable
{
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
     * The sentence tokenizer.
     *
     * @var \Rubix\ML\Other\Tokenizers\Sentence
     */
    protected $sentenceTokenizer;

    /**
     * @param int $n
     * @param int $skip
     * @param \Rubix\ML\Other\Tokenizers\Word|null $wordTokenizer
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $n = 2, int $skip = 2, ?Word $wordTokenizer = null)
    {
        if ($n < 2) {
            throw new InvalidArgumentException('Number of words per'
                . " token must be greater than 1, $n given.");
        }

        if ($skip < 0) {
            throw new InvalidArgumentException('Skip words must be'
                . " greater than 1, $skip given.");
        }

        $this->n = $n;
        $this->skip = $skip;
        $this->wordTokenizer = $wordTokenizer ?? new Word();
        $this->sentenceTokenizer = new Sentence();
    }

    /**
     * Tokenize a blob of text.
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array
    {
        $sentences = $this->sentenceTokenizer->tokenize($text);

        $skipGrams = [];

        foreach ($sentences as $sentence) {
            $words = $this->wordTokenizer->tokenize($sentence);

            $n = count($words);

            foreach ($words as $i => $word) {
                $p = min($n - ($i + $this->n), $this->skip);

                for ($j = 0; $j <= $p; ++$j) {
                    $skipGram = $word;

                    for ($k = 1; $k < $this->n; ++$k) {
                        $skipGram .= self::SEPARATOR . $words[$i + $j + $k];
                    }

                    $skipGrams[] = $skipGram;
                }
            }
        }

        return $skipGrams;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Skip Gram (n: {$this->n}, skip: {$this->skip}, word_tokenizer: {$this->wordTokenizer})";
    }
}
