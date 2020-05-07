<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function is_null;

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
     * @return \Rubix\ML\DataType[]
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if (!$dataset->homogeneous() or $dataset->columnType(0) != DataType::continuous()) {
            throw new InvalidArgumentException('This Transformer is'
                . ' only compatible with continuous data types.');
        }

        if (is_null($this->dfs) or is_null($this->n)) {
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
            $idfs[] = 1.0 + log($this->n / $df);
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
        if (is_null($this->idfs)) {
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
