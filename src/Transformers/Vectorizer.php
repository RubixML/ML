<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Math\Matrix;
use Rubix\Engine\Transformers\Tokenizers\Tokenizer;
use Rubix\Engine\Transformers\Tokenizers\WhitespaceTokenizer;

abstract class Vectorizer
{
    /**
     * @var \Rubix\Engine\Tokenizer
     */
    protected $tokenizer;

    /**
     * The vocabulary of the sampled data.
     *
     * @var array
     */
    protected $vocabulary = [
        //
    ];

    /**
     * @param  array  $data
     * @param  \Rubix\Engine\Tokenizer  $tokenizer
     * @return void
     */
    public function __construct(array $data = [], Tokenizer $tokenizer = null)
    {
        if (!isset($tokenizer)) {
            $tokenizer = new WhitespaceTokenizer();
        }

        $this->tokenizer = $tokenizer;

        foreach ($data as $sample) {
            foreach ($this->tokenizer->tokenize($sample) as $token) {
                if (!isset($this->vocabulary[$token])) {
                    $this->vocabulary[$token] = count($this->vocabulary);
                }
            }
        }
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
     * Transform an array of samples into an array of vectors.
     *
     * @param  array  $samples
     * @return array
     */
    public function transform(array $data) : array
    {
        return array_map(function ($sample) {
            return $this->vectorize($sample);
        }, $data);
    }

    /**
     * Convert a sample into a vector returning a fixed array of n length where
     * n is equal to the size of the vocabulary.
     *
     * @param  string  $sample
     * @return \Rubix\Engine\Math\Matrix
     */
    abstract public function vectorize(string $sample) : Matrix;
}
