<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
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
     * Calculate the inverse document frequency values for each feature column.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }

        $this->idfs = array_fill(0, $dataset->numColumns(), 0);

        foreach ($dataset as $sample) {
            foreach ($sample as $column => $feature) {
                if ($feature > 0) {
                    $this->idfs[$column]++;
                }
            }
        }

        foreach ($this->idfs as &$idf) {
            $idf = log($dataset->numRows() / ($idf + self::EPSILON), 10);
        }
    }

    /**
     * @return array
     */
    public function idfs() : ?array
    {
        return $this->idfs;
    }

    /**
     * Multiply the term frequency by the inverse document frequency.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (!isset($this->idfs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $i => &$feature) {
                $feature *= $this->idfs[$i];
            }
        }
    }
}
