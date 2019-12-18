<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

/**
 * TF-IDF Transformer
 *
 * Term Frequency - Inverse Document Frequency is a measure of how important
 * a word is to a document. The TF-IDF value increases proportionally with
 * the number of times a word appears in a document and is offset by the
 * frequency of the word in the corpus.
 *
 * > **Note**: This transformer assumes that its input is made up of word
 * frequency vectors such as those created by the Word Count Vectorizer.
 *
 * References:
 * [1] S. Robertson. (2003). Understanding Inverse Document Frequency: On
 * theoretical arguments for IDF.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TfIdfTransformer implements Transformer, Stateful, Elastic
{
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
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
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
     * Return the inverse document frequencies calculated during fitting.
     *
     * @return float[]|null
     */
    public function idfs() : ?array
    {
        return $this->idfs;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);
        
        $this->dfs = array_fill(0, $dataset->numColumns(), 1);
        $this->n = 1;

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if (!$dataset->homogeneous() or $dataset->columnType(0) !== DataType::CONTINUOUS) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        if ($this->dfs === null or $this->n === null) {
            $this->fit($dataset);
            
            return;
        }

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $column => $feature) {
                if ($feature > 0) {
                    ++$this->dfs[$column];
                }
            }
        }

        $this->n += $dataset->numRows();

        $idfs = [];

        foreach ($this->dfs as $df) {
            $idfs[] = log($this->n / $df) + 1.;
        }

        $this->idfs = $idfs;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->idfs === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$value) {
                if ($value > 0) {
                    $value *= $this->idfs[$column];
                }
            }
        }
    }
}
