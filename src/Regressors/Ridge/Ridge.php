<?php

namespace Rubix\ML\Regressors\Ridge;

use NDArray;
use NumPower;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_null;
use function array_fill;

/**
 * Ridge
 *
 * L2 regularized least squares linear model solved using a closed-form solution. The addition
 * of regularization, controlled by the *l2Penalty* parameter, makes Ridge less prone to overfitting
 * than ordinary linear regression.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Ridge implements Estimator, Learner, RanksFeatures, Persistable
{
    use AutotrackRevisions;

    /**
     * The strength of the L2 regularization penalty.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * The y intercept i.e. the bias added to the decision function.
     *
     * @var float|null
     */
    protected ?float $bias = null;

    /**
     * The computed coefficients of the regression line.
     *
     * @var NDArray|null
     */
    protected ?NDArray $coefficients = null;

    /**
     * The dimensionality of the training set.
     *
     * @var int<0,max>|null
     */
    protected ?int $featureCount = null;

    /**
     * @param float $l2Penalty
     * @throws InvalidArgumentException
     */
    public function __construct(float $l2Penalty = 1.0)
    {
        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException('L2 Penalty must be'
                . " greater than 0, $l2Penalty given.");
        }

        $this->l2Penalty = $l2Penalty;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the estimator is compatible with.
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
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'l2 penalty' => $this->l2Penalty,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->coefficients !== null and $this->bias !== null;
    }

    /**
     * Return the weights of features in the decision function.
     *
     * @return (int|float)[]|null
     */
    public function coefficients() : ?array
    {
        return $this->coefficients ? $this->coefficients->toArray() : null;
    }

    /**
     * Return the bias added to the decision function.
     *
     * @return float|null
     */
    public function bias() : ?float
    {
        return $this->bias;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $samples = $dataset->samples();

        $m = $dataset->numSamples();
        $n = $dataset->numFeatures();

        $xArr = [];

        foreach ($samples as $sample) {
            $xArr[] = array_merge([1.0], $sample);
        }

        $x = NumPower::array($xArr);
        $xT = NumPower::transpose($x, [1, 0]);

        $y = NumPower::reshape(NumPower::array($dataset->labels()), [$m, 1]);

        $p = $n + 1;

        $penalties = array_fill(0, $p, array_fill(0, $p, 0.0));

        for ($i = 1; $i < $p; ++$i) {
            $penalties[$i][$i] = $this->l2Penalty;
        }

        $penalties = NumPower::array($penalties);

        $xTx = NumPower::matmul($xT, $x);
        $xTxReg = NumPower::add($xTx, $penalties);
        $xTxInv = NumPower::inv($xTxReg);
        $xTy = NumPower::matmul($xT, $y);

        $beta = NumPower::matmul($xTxInv, $xTy);

        /** @var list<float> $betaArr */
        $betaArr = NumPower::reshape($beta, [$p])->toArray();

        $this->bias = $betaArr[0];
        $this->coefficients = NumPower::array(array_slice($betaArr, 1));
        $this->featureCount = $n;
    }

    /**
     * Make a prediction based on the line calculated from the training data.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->coefficients or is_null($this->bias) or is_null($this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $samples = NumPower::array($dataset->samples());
        $w = NumPower::reshape($this->coefficients, [$this->featureCount, 1]);

        $out = NumPower::matmul($samples, $w);
        $out = NumPower::add($out, $this->bias);

        /** @var list<int|float> */
        return NumPower::reshape($out, [$dataset->numSamples()])->toArray();
    }

    /**
     * Return the importance scores of each feature column of the training set.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (is_null($this->coefficients)) {
            throw new RuntimeException('Learner has not been trained.');
        }

        /** @var float[] */
        return NumPower::abs($this->coefficients)->toArray();
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
        return 'Ridge (' . Params::stringify($this->params()) . ')';
    }
}
