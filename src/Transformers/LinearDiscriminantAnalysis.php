<?php

namespace Rubix\ML\Transformers;

use NDArray;
use NumPower;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
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
use function count;

use const Rubix\ML\EPSILON;

/**
 * Linear Discriminant Analysis
 *
 * Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that
 * selects the most informative features based on their class labels. More formally, LDA finds
 * a linear combination of features that characterizes or best *discriminates* two or more
 * classes.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LinearDiscriminantAnalysis implements Transformer, Stateful, Persistable
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
        return isset($this->eigenvectors);
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
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Transformer requires a'
                . ' Labeled training set.');
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithTransformer($dataset, $this),
        ])->check();

        if ($dataset->labelType() != DataType::categorical()) {
            throw new InvalidArgumentException('Transformer requires'
                . " categorical labels, {$dataset->labelType()} given.");
        }

        [$m, $n] = $dataset->shape();

        $sW = NumPower::zeros([$n, $n], 'float32', 0);

        foreach ($dataset->stratifyByLabel() as $stratum) {
            $prior = $stratum->numSamples() / $m;

            $sW = NumPower::add(
                NumPower::multiply($this->covariance($stratum->samples()), $prior),
                $sW
            );
        }

        $eig = NumPower::eig(
            NumPower::subtract($this->covariance($dataset->samples()), $sW)
        );

        $eigenvalues = $eig[0]->toArray();
        $eigenvectors = array_map(null, ...$eig[1]->toArray());

        $totalVariance = array_sum($eigenvalues);

        array_multisort($eigenvalues, SORT_DESC, $eigenvectors);

        $eigenvalues = array_slice($eigenvalues, 0, $this->dimensions);
        $eigenvectors = array_slice($eigenvectors, 0, $this->dimensions);

        $eigenvectors = NumPower::array($eigenvectors, 'float32');
        $eigenvectors = NumPower::transpose($eigenvectors, [1, 0]);

        $noiseVariance = $totalVariance - array_sum($eigenvalues);
        $lossiness = $noiseVariance / ($totalVariance ?: EPSILON);

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
        if (!$this->eigenvectors) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = NumPower::matmul(
            NumPower::array($samples, 'float32'),
            $this->eigenvectors
        )->toArray();
    }

    /**
     * Compute the covariance matrix of a set of samples.
     *
     * @param list<list<mixed>> $samples
     * @return NDArray
     */
    protected function covariance(array $samples) : NDArray
    {
        $m = count($samples);

        $x = NumPower::array($samples, 'float32');

        $mean = NumPower::divide(NumPower::sum($x, axis: 0), $m);

        $centered = NumPower::subtract($x, $mean);

        return NumPower::divide(
            NumPower::matmul(NumPower::transpose($centered, [1, 0]), $centered),
            $m
        );
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
        return "Linear Discriminant Analysis (dimensions: {$this->dimensions})";
    }
}
