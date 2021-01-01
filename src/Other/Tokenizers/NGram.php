<?php

namespace Rubix\ML\Other\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

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
    /**
     * The separator between words in the n-gram.
     *
     * @var string
     */
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
     * @param int $min
     * @param int $max
     * @param \Rubix\ML\Other\Tokenizers\Word|null $wordTokenizer
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $min = 2, int $max = 2, ?Word $wordTokenizer = null)
    {
        if ($min < 1) {
            throw new InvalidArgumentException('Minimum cannot be less than 1.');
        }

        if ($max < $min) {
            throw new InvalidArgumentException('Maximum cannot be less than minimum.');
        }

        $this->min = $min;
        $this->max = $max;
        $this->wordTokenizer = $wordTokenizer ?? new Word();
        $this->sentenceTokenizer = new Sentence();
    }

    /**
     * Tokenize a blob of text.
     *
     * @internal
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array
    {
        $sentences = $this->sentenceTokenizer->tokenize($text);

        $nGrams = [];

        foreach ($sentences as $sentence) {
            $words = $this->wordTokenizer->tokenize($sentence);

            $n = count($words);

            foreach ($words as $i => $word) {
                $p = min($n - $i, $this->max);

                for ($j = $this->min; $j <= $p; ++$j) {
                    $nGram = $word;

                    for ($k = 1; $k < $j; ++$k) {
                        $nGram .= self::SEPARATOR . $words[$i + $k];
                    }

                    $nGrams[] = $nGram;
                }
            }
        }

        return $nGrams;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "N-Gram (min: {$this->min}, max: {$this->max}, word tokenizer: {$this->wordTokenizer})";
    }
}
