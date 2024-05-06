<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_slice;
use function array_multisort;
use function array_sum;

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
     * @var Matrix|null
     */
    protected ?Matrix $eigenvectors = null;

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
            new ExtensionIsLoaded('tensor'),
            new ExtensionMinimumVersion('tensor', '2.1.4'),
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

        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        if ($dataset->labelType() != DataType::categorical()) {
            throw new InvalidArgumentException('Transformer requires'
                . " categorical labels, {$dataset->labelType()} given.");
        }

        [$m, $n] = $dataset->shape();

        $sW = Matrix::zeros($n, $n);

        foreach ($dataset->stratifyByLabel() as $stratum) {
            $prior = $stratum->numSamples() / $m;

            $sW = Matrix::quick($stratum->samples())
                ->transpose()
                ->covariance()
                ->multiply($prior)
                ->add($sW);
        }

        $eig = Matrix::quick($dataset->samples())
            ->transpose()
            ->covariance()
            ->subtract($sW)
            ->eig(true);

        $eigenvalues = $eig->eigenvalues();
        $eigenvectors = $eig->eigenvectors()->asArray();

        $totalVariance = array_sum($eigenvalues);

        array_multisort($eigenvalues, SORT_DESC, $eigenvectors);

        $eigenvalues = array_slice($eigenvalues, 0, $this->dimensions);
        $eigenvectors = array_slice($eigenvectors, 0, $this->dimensions);

        $eigenvectors = Matrix::quick($eigenvectors)->transpose();

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

        $samples = Matrix::build($samples)
            ->matmul($this->eigenvectors)
            ->asArray();
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
