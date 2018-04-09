<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;
use Rubix\Engine\Preprocessors\Tokenizers\Tokenizer;
use Rubix\Engine\Preprocessors\Tokenizers\WhitespaceTokenizer;

class TokenCountVectorizer implements Preprocessor
{
    /**
     * The tokenizer used to extract text data into tokenable values.
     *
     * @var \Rubix\Engine\Transformers\Tokenizers\Tokenizer
     */
    protected $tokenizer;

    /**
     * The dictionary of stop words to filter out of the dataset.
     *
     * @var array
     */
    protected $stopWords = [
        //
    ];

    /**
     * The column types of the fitted dataset. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * The vocabulary of the fitted training set.
     *
     * @var array
     */
    protected $vocabulary = [
        //
    ];

    /**
     * @param  \Rubix\Engine\Preprocessors\Tokenizers\Tokenizer  $tokenizer
     * @param  array  $stopWords
     * @return void
     */
    public function __construct(Tokenizer $tokenizer = null, array $stopWords = [])
    {
        if (!isset($tokenizer)) {
            $tokenizer = new WhitespaceTokenizer();
        }

        $this->tokenizer = $tokenizer;
        $this->stopWords = array_fill_keys($stopWords, true);
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
     * @return array
     */
    public function stopWords() : array
    {
        return array_keys($this->stopWords);
    }

    /**
     * Build the vocabulary for the vectorizer.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        $this->columnTypes = $data->columnTypes();

        foreach ($data->samples() as $sample) {
            foreach ($sample as $column => $feature) {
                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $tokens = $this->tokenizer->tokenize($feature);

                    foreach ($tokens as $token) {
                        if (!isset($this->stopWords[$token])) {
                            if (!isset($this->vocabulary[$token])) {
                                $this->vocabulary[$token] = count($this->vocabulary);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Transform the text dataset into a collection of vectors where the value
     * is equal to the number of times that word appears in the sample.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as $column => $feature) {
                $vectors = [];

                if ($this->columnTypes[$column] === self::CATEGORICAL) {
                    $vectors[] = $this->vectorize($feature);
                }

                unset($sample[$column]);
            }

            $sample = array_merge(array_values($sample), ...$vectors);
        }
    }

    /**
     * Convert a string into a vector where the scalars are token counts.
     *
     * @param  string  $sample
     * @return array
     */
    public function vectorize(string $string) : array
    {
        $vector = array_fill_keys($this->vocabulary, 0);

        $tokens = $this->tokenizer->tokenize($string);

        foreach ($tokens as $token) {
            if (isset($this->vocabulary[$token])) {
                $vector[$this->vocabulary[$token]] += 1;
            }
        }

        return $vector;
    }
}
