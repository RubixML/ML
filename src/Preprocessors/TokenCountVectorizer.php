<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Preprocessors\Tokenizers\Tokenizer;
use Rubix\Engine\Preprocessors\Tokenizers\WhitespaceTokenizer;

class TokenCountVectorizer implements Preprocessor
{
    /**
     * The tokenizer used to extract text data.
     *
     * @var \Rubix\Engine\Transformers\Tokenizers\Tokenizer
     */
    protected $tokenizer;

    /**
     * The vocabulary of the sampled text data.
     *
     * @var array
     */
    protected $vocabulary = [
        //
    ];

    /**
     * @param  \Rubix\Engine\Tokenizer  $tokenizer
     * @return void
     */
    public function __construct(Tokenizer $tokenizer = null)
    {
        if (!isset($tokenizer)) {
            $tokenizer = new WhitespaceTokenizer();
        }

        $this->tokenizer = $tokenizer;
    }

    /**
     * @return array
     */
    public function vocabulary() : array
    {
        return array_flip($this->vocabulary);
    }

    /**
     * @return int
     */
    public function vocabularySize() : int
    {
        return count($this->vocabulary);
    }

    /**
     * Build the vocabulary for the vector encoder.
     *
     * @param  array  $samples
     * @param  array|null  $outcomes
     * @return void
     */
    public function fit(array $samples, ?array $outcomes = null) : void
    {
        foreach ($samples as $sample) {
            $tokens = $this->tokenizer->tokenize($sample[0]);

            foreach ($tokens as $token) {
                if (!isset($this->vocabulary[$token])) {
                    $this->vocabulary[$token] = count($this->vocabulary);
                }
            }
        }
    }

    /**
     * Transform an array of samples into an array of vectors.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $sample = $this->vectorize($sample[0]);
        }
    }

    /**
     * Convert a string into a vector where the values are token counts.
     *
     * @param  string  $sample
     * @return array
     */
    public function vectorize(string $sample) : array
    {
        $vector = array_fill_keys($this->vocabulary, 0);

        $tokens = $this->tokenizer->tokenize($sample);

        foreach ($tokens as $token) {
            if (isset($this->vocabulary[$token])) {
                $vector[$this->vocabulary[$token]] += 1;
            }
        }

        return $vector;
    }
}
