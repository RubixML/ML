<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\TracksRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_slice;

use const Rubix\ML\EPSILON;

/**
 * Truncated SVD
 *
 * Truncated Singular Value Decomposition (SVD) is a matrix factorization and dimensionality reduction technique that generalizes
 * eigendecomposition to general matrices. When applied to datasets of term frequency vectors, the technique is called Latent Semantic
 * Analysis (LSA) and computes a statistical model of relationships between words. Truncated SVD can also be used to compress document
 * representations for fast information retrieval and is known as Latent Semantic Indexing (LSI) in this context.
 *
 * References:
 * [1] P. W. Foltz. (1996) Latent semantic analysis for text-based research.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TruncatedSVD implements Transformer, Stateful, Persistable
{
    use TracksRevisions;

    /**
     * The target number of dimensions to project onto.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The component vectors of the singular value decomposition.
     *
     * @var \Tensor\Matrix|null
     */
    protected $components;

    /**
     * The percentage of information lost due to the transformation.
     *
     * @var float|null
     */
    protected $lossiness;

    /**
     * @param int $dimensions
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $dimensions)
    {
        if (!extension_loaded('tensor')) {
            throw new RuntimeException('Tensor extension not installed, check PHP configuration.');
        }

        if ($dimensions < 1) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        $this->dimensions = $dimensions;
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
        return isset($this->components);
    }

    /**
     * Return the percentage of information lost due to the transformation.
     *
     * @return float|null
     */
    public function lossiness() : ?float
    {
        return $this->lossiness;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $svd = Matrix::build($dataset->samples())->svd();

        $singularValues = $svd->singularValues();
        $vT = $svd->vT()->asArray();

        $totalVariance = array_sum($singularValues);

        array_multisort($singularValues, SORT_DESC, $vT);

        $singularValues = array_slice($singularValues, 0, $this->dimensions);
        $vT = array_slice($vT, 0, $this->dimensions);

        $components = Matrix::quick($vT)->transpose();

        $noiseVariance = $totalVariance - array_sum($singularValues);
        $lossiness = $noiseVariance / ($totalVariance ?: EPSILON);

        $this->components = $components;
        $this->lossiness = $lossiness;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!$this->components) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = Matrix::build($samples)
            ->matmul($this->components)
            ->asArray();
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Truncated SVD (dimensions: {$this->dimensions})";
    }
}
