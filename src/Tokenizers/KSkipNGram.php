<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function min;

/**
 * K-Skip-N-Gram
 *
 * K-skip-n-grams are a technique similar to n-grams, whereby n-grams are formed but
 * in addition to allowing adjacent sequences of words, the next *k* words will
 * be skipped forming n-grams of the new forward looking sequences. The tokenizer
 * outputs tokens ranging from *min* to *max* number of words per token.
 *
 * References:
 * [1] D. Guthrie et al. A Closer Look at Skip-gram Modelling.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Oksana Yudenko
 */
class KSkipNGram implements Tokenizer
{
    /**
     * The separator between words in the n-gram.
     *
     * @var string
     */
    protected const SEPARATOR = ' ';

    /**
     * The minimum number of words in a single token.
     *
     * @var int
     */
    protected int $min;

    /**
     * The maximum number of words in a single token.
     *
     * @var int
     */
    protected int $max;

    /**
     * The number of words to skip over to form new sequences.
     *
     * @var int
     */
    protected int $skip;

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
     * @param int $skip
     * @param Word|null $wordTokenizer
     * @throws InvalidArgumentException
     */
    public function __construct(int $min = 2, int $max = 2, int $skip = 2, ?Word $wordTokenizer = null)
    {
        if ($min < 1) {
            throw new InvalidArgumentException('Minimum cannot be less than 1.');
        }

        if ($max < $min) {
            throw new InvalidArgumentException('Maximum cannot be less than minimum.');
        }

        if ($skip < 0) {
            throw new InvalidArgumentException('Skip words must be'
                . " greater than 1, $skip given.");
        }

        $this->min = $min;
        $this->max = $max;
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
                for ($j = $this->min; $j <= $this->max; ++$j) {
                    $p = min($n - ($i + $j), $this->skip);

                    for ($k = 0; $k <= $p; ++$k) {
                        $skipGram = $word;

                        for ($l = 1; $l < $j; ++$l) {
                            $skipGram .= self::SEPARATOR . $words[$i + $k + $l];
                        }

                        $skipGrams[] = $skipGram;
                    }
                }
            }
        }

        return $skipGrams;
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
        return "K-Skip-N-Gram (min: {$this->min}, max: {$this->max}, skip: {$this->skip}, word tokenizer: {$this->wordTokenizer})";
    }
}
