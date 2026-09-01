<?php

namespace Rubix\ML\Transformers;

use NDArray;
use NumPower;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_map;
use function array_slice;
use function array_multisort;
use function array_sum;

use const Rubix\ML\EPSILON;

/**
 * Principal Component Analysis
 *
 * Principal Component Analysis or *PCA* is a dimensionality reduction technique that
 * aims to transform the feature space by the *k* principal components that explain
 * the most variance of the data where *k* is the dimensionality of the output
 * specified by the user. PCA is used to compress high dimensional samples down to
 * lower dimensions such that would retain as much of the information within the data
 * as possible.
 *
 * References:
 * [1] H. Abdi et al. (2010). Principal Component Analysis.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PrincipalComponentAnalysis implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The target number of dimensions to project onto.
     *
     * @var int
     */
    protected int $dimensions;

    /**
     * The matrix of eigenvectors computed at fitting.
     *
     * @var NDArray|null
     */
    protected ?NDArray $eigenvectors = null;

    /**
     * The percentage of information lost due to the transformation.
     *
     * @var float|null
     */
    protected ?float $lossiness = null;

    /**
     * The centers (means) of the input feature columns.
     *
     * @var NDArray|null
     */
    protected ?NDArray $mean = null;

    /**
     * @param int $dimensions
     * @throws InvalidArgumentException
     */
    public function __construct(int $dimensions)
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

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
     * @return list<DataType>
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
        return $this->mean and $this->eigenvectors;
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
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithTransformer($dataset, $this),
        ])->check();

        $samples = $dataset->samples();

        $m = count($samples);

        $x = NumPower::array($samples, 'float32');

        $mean = NumPower::divide(NumPower::sum($x, axis: 0), $m);

        $centered = NumPower::subtract($x, $mean);

        $covariance = NumPower::divide(
            NumPower::matmul(NumPower::transpose($centered, [1, 0]), $centered),
            $m
        );

        $eig = NumPower::eig($covariance);

        $eigenvalues = $eig[0]->toArray();
        $eigenvectors = array_map(null, ...$eig[1]->toArray());

        $totalVariance = array_sum($eigenvalues);

        array_multisort($eigenvalues, SORT_DESC, $eigenvectors);

        $eigenvalues = array_slice($eigenvalues, 0, $this->dimensions);
        $eigenvectors = array_slice($eigenvectors, 0, $this->dimensions);

        $eigenvectors = NumPower::array($eigenvectors, $eig[1]->dataType());
        $eigenvectors = NumPower::transpose($eigenvectors, [1, 0]);

        $noiseVariance = $totalVariance - array_sum($eigenvalues);
        $lossiness = $noiseVariance / ($totalVariance ?: EPSILON);

        $this->mean = $mean;
        $this->eigenvectors = $eigenvectors;
        $this->lossiness = $lossiness;
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!$this->mean or !$this->eigenvectors) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $x = NumPower::array($samples, $this->mean->dataType());

        $delta = NumPower::subtract($x, $this->mean);

        $samples = NumPower::matmul($delta, $this->eigenvectors)->toArray();
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Principal Component Analysis (dimensions: {$this->dimensions})";
    }
}
