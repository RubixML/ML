<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('L2 regularization term must'
                . ' be non-negative.');
        }

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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $coefficients = $this->computeCoefficients($dataset->samples(),
            $dataset->labels());

        $this->intercept = array_shift($coefficients);
        $this->coefficients = $coefficients;
    }

    /**
     * Make a prediction based on the line calculated from the training data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
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

            $predictions[] = $outcome;
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
