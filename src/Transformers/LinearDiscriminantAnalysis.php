<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\warn_deprecated;
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
    protected $dimensions;

    /**
     * The matrix of eigenvectors computed at fitting.
     *
     * @var \Tensor\Matrix|null
     */
    protected $eigenvectors;

    /**
     * The amount of variance that is preserved by the transformation.
     *
     * @var float|null
     */
    protected $explainedVar;

    /**
     * The amount of variance lost by discarding the noise components.
     *
     * @var float|null
     */
    protected $noiseVar;

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
     * Return the amount of variance that has been preserved by the transformation.
     *
     * @deprecated
     *
     * @return float|null
     */
    public function explainedVar() : ?float
    {
        warn_deprecated('ExplainedVar() is deprecated, use lossiness() instead.');

        return $this->explainedVar;
    }

    /**
     * Return the amount of variance lost by discarding the noise components.
     *
     * @deprecated
     *
     * @return float|null
     */
    public function noiseVar() : ?float
    {
        warn_deprecated('NoiseVar() is deprecated, use lossiness() instead.');

        return $this->noiseVar;
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

        foreach ($dataset->stratify() as $stratum) {
            $sW = Matrix::build($stratum->samples())
                ->transpose()
                ->covariance()
                ->multiply($stratum->numRows() / $m)
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

        $explainedVariance = array_sum($eigenvalues);
        $noiseVariance = $totalVariance - $explainedVariance;
        $lossiness = $noiseVariance / ($totalVariance ?: EPSILON);

        $this->explainedVar = $explainedVariance;
        $this->noiseVar = $noiseVariance;
        $this->lossiness = $lossiness;

        $this->eigenvectors = $eigenvectors;
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
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
     * @return string
     */
    public function __toString() : string
    {
        return "Linear Discriminant Analysis (dimensions: {$this->dimensions})";
    }
}
