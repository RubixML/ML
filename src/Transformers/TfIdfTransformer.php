<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_fill;
use function array_sum;
use function log;

/**
 * TF-IDF Transformer
 *
 * Term Frequency - Inverse Document Frequency is a measure of how important a word is to
 * a document. The TF-IDF value increases with the number of times a word appears in a document
 * and is offset by the frequency of the word in the corpus.
 *
 * > **Note**: TF-IDF Transformer assumes that its input is made up of term frequency
 * vectors such as those created by Word Count Vectorizer.
 *
 * References:
 * [1] S. Robertson. (2003). Understanding Inverse Document Frequency: On theoretical
 * arguments for IDF.
 * [2] S. Robertson et al. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
 * [3] C. D. Manning et al. (2009). An Introduction to Information Retrieval.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TfIdfTransformer implements Transformer, Stateful, Elastic, Persistable
{
    use AutotrackRevisions;

    /**
     * The amount of additive (Laplace) smoothing to add to the IDFs.
     *
     * @var float
     */
    protected $smoothing;

    /**
     * Should we apply a sub-linear function to dampen the effect of recurring tokens?
     *
     * @var bool
     */
    protected $dampening;

    /**
     * Should we normalize by document length?
     *
     * @var bool
     */
    protected $normalize;

    /**
     * The document frequencies of each word i.e. the number of times a word appeared in a document.
     *
     * @var int[]|null
     */
    protected $dfs;

    /**
     * The inverse document frequencies for each feature column.
     *
     * @var float[]|null
     */
    protected $idfs;

    /**
     * The number of tokens fitted so far.
     *
     * @var int|null
     */
    protected $tokenCount;

    /**
     * The number of documents (samples) that have been fitted so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * The average token count per document.
     *
     * @var float|null
     */
    protected $averageDocumentLength;

    /**
     * @param float $smoothing
     * @param bool $dampening
     * @param bool $normalize
     */
    public function __construct(float $smoothing = 1.0, bool $dampening = false, bool $normalize = false)
    {
        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        $this->smoothing = $smoothing;
        $this->dampening = $dampening;
        $this->normalize = $normalize;
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
        return isset($this->idfs) and isset($this->averageDocumentLength);
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
     * Return the average length of a document in tokens.
     *
     * @return float|null
     */
    public function averageDocumentLength() : ?float
    {
        return $this->averageDocumentLength;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $this->dfs = array_fill(0, $dataset->numFeatures(), 0);
        $this->tokenCount = $this->n = 0;

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
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithTransformer($dataset, $this),
        ])->check();

        if ($this->dfs === null) {
            $this->fit($dataset);

            return;
        }

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $column => $value) {
                if ($value > 0) {
                    ++$this->dfs[$column];

                    $this->tokenCount += $value;
                }
            }
        }

        $this->n += $dataset->numSamples();

        $this->averageDocumentLength = $this->tokenCount / $this->n;

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
        if ($this->idfs === null or $this->averageDocumentLength === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            if ($this->normalize) {
                $documentLength = array_sum($sample);

                if ($documentLength == 0) {
                    continue;
                }

                $delta = $this->averageDocumentLength / $documentLength;
            }

            foreach ($sample as $column => &$value) {
                if ($value > 0) {
                    if (isset($delta)) {
                        $value *= $delta;
                    }

                    if ($this->dampening) {
                        $value = 1.0 + log($value);
                    }

                    $value *= $this->idfs[$column];
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
        return "TF-IDF Transformer (smoothing: {$this->smoothing}, dampening: "
            . Params::toString($this->dampening) . ', normalize: '
            . Params::toString($this->normalize) . ')';
    }
}
