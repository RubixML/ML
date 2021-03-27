<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_null;

/**
 * TF-IDF Transformer
 *
 * Term Frequency - Inverse Document Frequency is a measure of how important a word is to
 * a document. The TF-IDF value increases proportionally (linearly) with the number of
 * times a word appears in a document and is offset by the frequency of the word in the
 * corpus.
 *
 * > **Note**: TF-IDF Transformer assumes that its input is made up of term frequency
 * vectors such as those created by Word Count Vectorizer.
 *
 * References:
 * [1] S. Robertson. (2003). Understanding Inverse Document Frequency: On
 * theoretical arguments for IDF.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TfIdfTransformer implements Transformer, Stateful, Elastic, Persistable
{
    use AutotrackRevisions;

    /**
     * The amount of additive (Laplace) smoothing to add to the inverse document
     * frequencies (IDFs).
     *
     * @var float
     */
    protected $smoothing;

    /**
     * The document frequencies of each word i.e. the number of times a word
     * appeared in a document given the entire corpus.
     *
     * @var int[]|null
     */
    protected $dfs;

    /**
     * The inverse document frequency values for each feature column.
     *
     * @var float[]|null
     */
    protected $idfs;

    /**
     * The number of documents (samples) that have been fitted so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * @param float $smoothing
     */
    public function __construct(float $smoothing = 1.0)
    {
        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        $this->smoothing = $smoothing;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->idfs);
    }

    /**
     * Return the document frequencies calculated during fitting.
     *
     * @return int[]|null
     */
    public function dfs() : ?array
    {
        return $this->dfs;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $this->dfs = array_fill(0, $dataset->numColumns(), 0);
        $this->n = 0;

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function update(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        if (is_null($this->dfs) or is_null($this->n)) {
            $this->fit($dataset);

            return;
        }

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $column => $value) {
                if ($value > 0) {
                    ++$this->dfs[$column];
                }
            }
        }

        $this->n += $dataset->numRows();

        $nHat = $this->n + $this->smoothing;

        $idfs = [];

        foreach ($this->dfs as $df) {
            $idfs[] = 1.0 + log($nHat / ($df + $this->smoothing));
        }

        $this->idfs = $idfs;
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->idfs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$tf) {
                if ($tf > 0) {
                    $tf *= $this->idfs[$column];
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "TF-IDF Transformer (smoothing: {$this->smoothing})";
    }
}
