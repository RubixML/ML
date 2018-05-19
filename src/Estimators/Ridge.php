<?php

namespace Rubix\Engine\Estimators;

use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\Matrix;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Supervised;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\Engine\Estimators\Persistable;
use Rubix\Engine\Estimators\Predictions\Prediction;
use InvalidArgumentException;

class Ridge implements Regressor, Persistable
{
    /**
     * The regularization parameter that controls the penalty to the size of the
     * coeffecients. i.e. the ridge penalty.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The computed y intercept.
     *
     * @var float
     */
    protected $intercept;

    /**
     * The computed coefficients of the training data.
     *
     * @var array
     */
    protected $coefficients = [
        //
    ];

    /**
     * @param  float  $alpha
     * @return void
     */
    public function __construct(float $alpha = 1.0)
    {
        $this->alpha = $alpha;
    }

    /**
     * @return float|null
     */
    public function intercept() : ?float
    {
        return $this->intercept;
    }

    /**
     * @return array
     */
    public function coefficients() : array
    {
        return $this->coefficients;
    }

    /**
     * Calculate the coefficients of the training data. i.e. compute the line
     * that best fits the training data.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous samples.');
        }

        $coefficients = $this->computeCoefficients(...$dataset->all());

        $this->intercept = array_shift($coefficients);
        $this->coefficients = $coefficients;
    }

    /**
     * Make a prediction based on the line calculated from the training data.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
            $outcome = $this->intercept;

            foreach ($this->coefficients as $column => $coefficient) {
                $outcome += $coefficient * $sample[$column];
            }

            $predictions[] = new Prediction($outcome);
        }

        return $predictions;
    }

    /**
     * Compute the coefficients of the training data like ordinary least squares,
     * however add a regularization term to the equation.
     *
     * @param  array  $samples
     * @param  array  $labels
     * @return array
     */
    protected function computeCoefficients(array $samples, array $labels) : array
    {
        foreach ($samples as &$sample) {
            array_unshift($sample, 1);
        }

        $x = new Matrix($samples);
        $y = new Vector($labels);
        $a = MatrixFactory::identity($x->getN())->scalarMultiply($this->alpha);

        return $x->transpose()->multiply($x)->add($a)->inverse()
            ->multiply($x->transpose()->multiply($y))
            ->getColumn(0);
    }
}
