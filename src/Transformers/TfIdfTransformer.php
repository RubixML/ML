<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
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
     * @var \Rubix\Tensor\Vector|null
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
     * @return \Rubix\Tensor\Vector|null
     */
    public function idfs() : ?Vector
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
        $this->counts = Vector::ones($dataset->numColumns())->asArray();
        $this->n = 1;

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
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }
        
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

        $idfs = [];

        foreach ($this->counts as $column => $count) {
            $idfs[] = log($this->n / $count) + 1.;
        }

        $this->idfs = Vector::quick($idfs);
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

        $samples = Matrix::quick($samples)
            ->multiply($this->idfs)
            ->asArray();
    }
}
