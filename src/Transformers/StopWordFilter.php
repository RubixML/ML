<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;

class StopWordFilter implements Transformer
{
    /**
     * The dictionary of stop words to filter out of the dataset.
     *
     * @var array
     */
    protected $stopWords = [
        //
    ];

    /**
     * @param  array  $stopWords
     * @return void
     */
    public function __construct(array $stopWords = [])
    {
        $this->stopWords = array_values($stopWords);
    }

    /**
     * Build the vocabulary for the vectorizer.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        //
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
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    $feature = preg_replace('/\b('
                        . implode('|', $this->stopWords)
                        . ')\b/', '', $feature);
                }
            }
        }
    }
}
