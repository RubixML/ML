<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function min;

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
    protected int $min;

    /**
     * The maximum number of contiguous words in a single token.
     *
     * @var int
     */
    protected int $max;

    /**
     * The word tokenizer.
     *
     * @var Word
     */
    protected Word $wordTokenizer;

    /**
     * The sentence tokenizer.
     *
     * @var Sentence
     */
    protected Sentence $sentenceTokenizer;

    /**
     * @param int $min
     * @param int $max
     * @param Word|null $wordTokenizer
     * @throws InvalidArgumentException
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "N-Gram (min: {$this->min}, max: {$this->max}, word tokenizer: {$this->wordTokenizer})";
    }
}
