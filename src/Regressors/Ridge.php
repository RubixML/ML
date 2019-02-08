<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\Tensor\Vector;
use Rubix\Tensor\Matrix;
use Rubix\ML\Persistable;
use Rubix\Tensor\ColumnVector;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Ridge
 *
 * L2 penalized least squares regression. Can be used for simple regression
 * problems that can be fit to a straight line.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ridge implements Learner, Persistable
{
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
     * @var \Rubix\Tensor\Vector|null
     */
    protected $weights;

    /**
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1.)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization penalty must'
                . " 0 or greater, $alpha given.");
        }

        $this->alpha = $alpha;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
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
     * @return array|null
     */
    public function weights() : ?array
    {
        return isset($this->weights) ? $this->weights->asArray() : null;
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
     * Calculate the coefficients of the training data. i.e. compute the line
     * that best fits the training data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $samples = $dataset->samples();
        $labels = $dataset->labels();

        $biases = Matrix::ones($dataset->numRows(), 1);

        $x = Matrix::build($samples)->augmentLeft($biases);
        $y = Vector::build($labels);

        $alphas = array_fill(0, $x->n() - 1, $this->alpha);

        $penalty = Matrix::diagonal(array_merge([0.], $alphas));

        $xT = $x->transpose();

        $coefficients = $xT->matmul($x)
            ->add($penalty)
            ->inverse()
            ->dot($xT->dot($y))
            ->asArray();

        $this->bias = (float) array_shift($coefficients);
        $this->weights = Vector::quick($coefficients);
    }

    /**
     * Make a prediction based on the line calculated from the training data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (is_null($this->weights) or is_null($this->bias)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return Matrix::build($dataset->samples())
            ->dot($this->weights)
            ->add($this->bias)
            ->asArray();
    }
}
