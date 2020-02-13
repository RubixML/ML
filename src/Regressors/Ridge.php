<?php

namespace Rubix\ML\Regressors;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Ridge
 *
 * L2 penalized ordinary least squares linear regression (OLS) solved using the closed-form
 * equation. The addition of regularization, controlled by the *alpha* parameter, makes Ridge
 * less prone to overfitting than non-regularized linear regression.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ridge implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The regularization parameter that controls the penalty to the size of the
     * coeffecients. i.e. the ridge penalty.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The y intercept.
     *
     * @var float|null
     */
    protected $bias;

    /**
     * The computed coefficients of the regression line.
     *
     * @var \Tensor\Vector|null
     */
    protected $weights;

    /**
     * @param float $alpha
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " 0 or greater, $alpha given.");
        }

        $this->alpha = $alpha;
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that this estimator is compatible with.
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
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'alpha' => $this->alpha,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->bias and $this->weights;
    }

    /**
     * Return the weights of the model.
     *
     * @return (int|float)[]|null
     */
    public function weights() : ?array
    {
        return $this->weights ? $this->weights->asArray() : null;
    }

    /**
     * Return the bias parameter of the regression line.
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $biases = Matrix::ones($dataset->numRows(), 1);

        $x = Matrix::build($dataset->samples())->augmentLeft($biases);
        $y = Vector::build($dataset->labels());

        $alphas = array_fill(0, $x->n() - 1, $this->alpha);

        array_unshift($alphas, 0.0);

        $penalties = Matrix::diagonal($alphas);

        $xT = $x->transpose();

        $coefficients = $xT->matmul($x)
            ->add($penalties)
            ->inverse()
            ->dot($xT->dot($y))
            ->asArray();

        $this->bias = (float) array_shift($coefficients);
        $this->weights = Vector::quick($coefficients);
    }

    /**
     * Make a prediction based on the line calculated from the training data.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->weights or $this->bias === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return Matrix::build($dataset->samples())
            ->dot($this->weights)
            ->add($this->bias)
            ->asArray();
    }
}
