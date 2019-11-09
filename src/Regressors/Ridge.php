<?php

namespace Rubix\ML\Regressors;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Ridge
 *
 * L2 penalized least squares linear regression solved using the closed-form
 * equation. The addition of regularization controlled by the alpha parameter
 * makes Ridge less prone to overfitting than ordinary least squares linear
 * regression.
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
    public function __construct(float $alpha = 1.)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('Alpha must be'
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

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $biases = Matrix::ones($dataset->numRows(), 1);

        $x = Matrix::build($dataset->samples())->augmentLeft($biases);
        $y = Vector::build($dataset->labels());

        $alphas = array_fill(0, $x->n() - 1, $this->alpha);

        $penalties = Matrix::diagonal(array_merge([0.], $alphas));

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
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->weights or $this->bias === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return Matrix::build($dataset->samples())
            ->dot($this->weights)
            ->add($this->bias)
            ->asArray();
    }
}
