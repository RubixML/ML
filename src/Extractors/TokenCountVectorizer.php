<?php

namespace Rubix\Engine\Extractors;

use Rubix\Engine\Extractors\Tokenizers\Word;
use Rubix\Engine\Extractors\Tokenizers\Tokenizer;
use InvalidArgumentException;
use RuntimeException;

class TokenCountVectorizer implements Extractor
{
    /**
     * The maximum size of the vocabulary.
     *
     * @var int
     */
    protected $maxVocabulary;

    /**
     * The tokenizer used to extract text data into tokenable values.
     *
     * @var \Rubix\Engine\Extractors\Tokenizers\Tokenizer
     */
    protected $tokenizer;

    /**
     * The vocabulary of the fitted training set.
     *
     * @var array
     */
    protected $vocabulary = [
        //
    ];

    /**
     * @param  int  $maxVocabulary
     * @param  \Rubix\Engine\Extractors\Tokenizers\Tokenizer  $tokenizer
     * @return void
     */
    public function __construct(int $maxVocabulary = PHP_INT_MAX, Tokenizer $tokenizer = null)
    {
        if ($maxVocabulary < 1) {
            throw new InvalidArgumentException('The size of the vocabulary must'
                . ' be at least 1 word.');
        }

        if (!isset($tokenizer)) {
            $tokenizer = new Word();
        }

        $this->maxVocabulary = $maxVocabulary;
        $this->tokenizer = $tokenizer;
    }

    /**
     * Return an array of words in the vocabulary.
     *
     * @return array
     */
    public function vocabulary() : array
    {
        return array_flip($this->vocabulary);
    }

    /**
     * Return the size of the vocabulary.
     *
     * @return int
     */
    public function size() : int
    {
        return count($this->vocabulary);
    }

    /**
     * Build the vocabulary for the vectorizer.
     *
     * @param  array  $samples
     * @return void
     */
    public function fit(array $samples) : void
    {
        $this->vocabulary = $frequencies = [];

        foreach ($samples as $sample) {
            if (is_string($sample)) {
                foreach ($this->tokenizer->tokenize($sample) as $token) {
                    if (isset($frequencies[$token])) {
                        $frequencies[$token]++;
                    } else {
                        $frequencies[$token] = 1;
                    }
                }
            }
        }

        if (count($frequencies) > $this->maxVocabulary) {
            arsort($frequencies);

            $frequencies = array_splice($frequencies, 0, $this->maxVocabulary);
        }

        foreach ($frequencies as $token => $count) {
            $this->vocabulary[$token] = count($this->vocabulary);
        }
    }

    /**
     * Transform the text dataset into a collection of vectors where the value
     * is equal to the number of times that word appears in the sample.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return array
     */
    public function extract(array $samples) : array
    {
        if (empty($this->vocabulary)) {
            throw new RuntimeException('Vocabulary is empty, try fitting a'
                . ' dataset first.');
        }

        $vectors = [];

        foreach ($samples as $sample) {
            if (is_string($sample)) {
                $vectors[] = $this->vectorize($sample);
            }
        }

        return $vectors;
    }

    /**
     * Convert a string into a vector where the scalars are token frequencies.
     *
     * @param  string  $sample
     * @return array
     */
    public function vectorize(string $sample) : array
    {
        $vector = array_fill(0, count($this->vocabulary), 0);

        foreach ($this->tokenizer->tokenize($sample) as $token) {
            if (isset($this->vocabulary[$token])) {
                $vector[$this->vocabulary[$token]] += 1;
            }
        }

        return $vector;
    }
}
