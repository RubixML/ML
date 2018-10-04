<?php

namespace Rubix\ML\Transformers;

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
class TfIdfTransformer implements Transformer
{
    /**
     * The inverse document frequency values for each feature.
     *
     * @var array|null
     */
    protected $idfs;

    /**
     * Return the inverse document frequencies calculated during fitting.
     *
     * @return array
     */
    public function idfs() : ?array
    {
        return $this->idfs;
    }

    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataframe->types())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }

        list($m, $n) = $dataframe->shape();

        $this->idfs = array_fill(0, $n, 0.);

        foreach ($dataframe as $sample) {
            foreach ($sample as $column => $feature) {
                if ($feature > 0) {
                    $this->idfs[$column]++;
                }
            }
        }

        foreach ($this->idfs as &$idf) {
            $idf = log(($idf !== 0. ? $m / $idf : 1.), 10);
        }
    }

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
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
