<?php

namespace Rubix\ML\Regressors;

use NDArray;
use NumPower;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Traits\AutotrackRevisions;
use function is_array;
use function is_float;
use function is_null;
use function Rubix\ML\array_pack;

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
        return $this->coefficients and isset($this->bias);
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
     * Train the learner with a dataset using NumPower for the algebra path.
     * Formula: (Xᵀ X + λ I)⁻¹ Xᵀ y
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

        $biases = NumPower::ones([$dataset->numSamples(), 1]);

        $samples = NumPower::array(array_pack($dataset->samples()));
        // Add bias from left
        $x = NumPower::concatenate([$biases, $samples], axis: 1);
        $y = NumPower::array($dataset->labels());

        /** @var int<0,max> $nHat */
        $nHat = $x->shape()[1] - 1;

        $penalties = array_fill(0, $nHat, $this->l2Penalty);
        array_unshift($penalties, 0.0);

        $penalties = NumPower::diag($penalties);

        $xT = NumPower::transpose($x, [1, 0]);

        $a = NumPower::add(NumPower::matmul($xT, $x), $penalties);
        $b = NumPower::dot($xT, $y);

        $coefficients = NumPower::dot(NumPower::inv($a), $b)->toArray();

        $this->bias = (float) array_shift($coefficients);
        $this->coefficients = NumPower::array($coefficients);
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
        if (!$this->coefficients or is_null($this->bias)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $weights = $this->coefficients->toArray();

        DatasetHasDimensionality::with($dataset, count($weights))->check();

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $x = NumPower::array($sample);
            $dot = NumPower::dot($x, $this->coefficients);
            $result = NumPower::add($dot, $this->bias);

            if (is_float($result)) {
                $predictions[] = $result;

                continue;
            }

            $value = $result->toArray();

            if (is_array($value)) {
                $value = $value[0] ?? null;
            }

            $predictions[] = (float) $value;
        }

        return $predictions;
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
