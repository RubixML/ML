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
     * The column types of the fitted dataset. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
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
        $this->columnTypes = $dataset->columnTypes();
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
            foreach ($this->columnTypes as $column => $type) {
                if ($type === self::CATEGORICAL) {
                    $sample[$column] = preg_replace('/\b('. implode('|', $this->stopWords) . ')\b/', '', $sample[$column]);
                }
            }
        }
    }
}
