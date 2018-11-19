<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;
use RuntimeException;

/**
 * TF-IDF Transformer
 *
 * Term Frequency - Inverse Document Frequency is the measure of how important a
 * word is to a document. The TF-IDF value increases proportionally with the
 * number of times a word appears in a document and is offset by the frequency
 * of the word in the corpus. This Transformer makes the assumption that the
 * input is made up of word frequency vectors such as those created by the Count
 * Vectorizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TfIdfTransformer implements Elastic
{
    /**
     * The times a word / feature appeared in a document.
     *
     * @var array|null
     */
    protected $counts;

    /**
     * The inverse document frequency values for each feature column.
     *
     * @var array|null
     */
    protected $idfs;

    /**
     * The number of documents (samples) that have been fitted so far.
     * 
     * @var int
     */
    protected $n;

    /**
     * Return the document counts for each word (feature column).
     *
     * @return array|null
     */
    public function counts() : ?array
    {
        return $this->counts;
    }

    /**
     * Return the inverse document frequencies calculated during fitting.
     *
     * @return array|null
     */
    public function idfs() : ?array
    {
        return $this->idfs;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $this->counts = array_fill(0, $dataset->numColumns(), 0);
        $this->n = 0;

        $this->idfs = [];

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function update(Dataset $dataset) : void
    {
        if (is_null($this->counts)) {
            $this->fit($dataset);
            return;
        }

        foreach ($dataset as $sample) {
            foreach ($sample as $column => $feature) {
                if ($feature > 0) {
                    $this->counts[$column]++;
                }
            }
        }

        $this->n += $dataset->numRows();

        foreach ($this->counts as $column => $count) {
            $idf = log($this->n / ($count ?: self::EPSILON), 10);

            $this->idfs[$column] = $idf;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        if (is_null($this->idfs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $i => &$feature) {
                $feature *= $this->idfs[$i];
            }
        }
    }
}
