<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_fill;
use function array_sum;
use function log;

/**
 * BM25 Transformer
 *
 * BM25 is a sublinear term weighting scheme that takes term frequency (TF), document frequency (DF),
 * and document length into account. It is similar to [TF-IDF](tf-idf-transformer.md) but with variable
 * sublinearity and the addition of document length normalization.
 *
 * > **Note**: BM25 Transformer assumes that its inputs are made up of token frequency
 * vectors such as those created by the Word Count or Token Hashing Vectorizer.
 *
 * References:
 * [1] S. Robertson et al. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
 * [2] K. Sparck Jones et al. (2000). A probabilistic model of information retrieval:
 * development and comparative experiments.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BM25Transformer implements Transformer, Stateful, Elastic
{
    /**
     * The term frequency (TF) dampening factor i.e. the `K1` parameter in the formula.
     * Lower values will cause the TF to saturate quicker.
     *
     * @var float
     */
    protected float $dampening;

    /**
     * The importance of document length in normalizing the term frequency i.e. the `b`
     * parameter in the formula
     *
     * @var float
     */
    protected float $normalization;

    /**
     * The document frequencies of each word i.e. the number of times a word appeared in
     * a document given the entire corpus.
     *
     * @var int[]|null
     */
    protected ?array $dfs = null;

    /**
     * The inverse document frequency values for each feature column.
     *
     * @var float[]|null
     */
    protected ?array $idfs = null;

    /**
     * The number of tokens fitted so far.
     *
     * @var int|null
     */
    protected ?int $totalTokens;

    /**
     * The number of documents (samples) that have been fitted so far.
     *
     * @var int|null
     */
    protected ?int $n;

    /**
     * The average token count per document.
     *
     * @var float|null
     */
    protected ?float $averageDocumentLength;

    /**
     * @param float $dampening
     * @param float $normalization
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $dampening = 1.2, float $normalization = 0.75)
    {
        if ($dampening < 0.0) {
            throw new InvalidArgumentException('Dampening must be greater'
                . " than 0, $dampening given.");
        }

        if ($normalization < 0.0 or $normalization > 1.0) {
            throw new InvalidArgumentException('Normalization must be between'
                . " 0 and 1, $normalization given.");
        }

        $this->dampening = $dampening;
        $this->normalization = $normalization;
    }

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
        return $this->idfs and $this->averageDocumentLength;
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
     * Return the average number of tokens per document.
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
        $this->dfs = array_fill(0, $dataset->numFeatures(), 1);
        $this->totalTokens = 0;
        $this->n = 1;

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

        if ($this->dfs === null or $this->n === null) {
            $this->fit($dataset);

            return;
        }

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $column => $tf) {
                if ($tf > 0) {
                    ++$this->dfs[$column];

                    $this->totalTokens += $tf;
                }
            }
        }

        $this->n += $dataset->numSamples();

        $this->averageDocumentLength = $this->totalTokens / $this->n;

        $idfs = [];

        foreach ($this->dfs as $df) {
            $idfs[] = log(1.0 + ($this->n - $df + 0.5) / ($df + 0.5));
        }

        $this->idfs = $idfs;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<array<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->idfs === null or $this->averageDocumentLength === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $delta = array_sum($sample) / $this->averageDocumentLength;

            $delta = 1.0 - $this->normalization + $this->normalization * $delta;

            $delta *= $this->dampening;

            foreach ($sample as $column => &$tf) {
                if ($tf > 0) {
                    $tf /= $tf + $delta;
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
        return "BM25 Transformer (dampening: {$this->dampening}, normalization: {$this->normalization})";
    }
}
