<?php

namespace Rubix\Engine;

use MathPHP\LinearAlgebra\Vector;
use Rubix\Engine\Datasets\Supervised;
use MathPHP\LinearAlgebra\MatrixFactory;
use MathPHP\LinearAlgebra\DiagonalMatrix;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;

class Ridge implements Estimator, Regressor, Persistable
{
    /**
     * The regularization parameter that controls the penalty to the size of the
     * coeffecients.
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
     * Learn the coefficients of the training data. i.e. compute the line that best
     * fits the training data.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $coefficients = $this->computeCoefficients($dataset->samples(), $dataset->outcomes());

        $this->intercept = array_shift($coefficients);
        $this->coefficients = $coefficients;
    }

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $outcome = $this->intercept;

        foreach ($this->coefficients as $column => $coefficient) {
            $outcome += $coefficient * $sample[$column];
        }

        return new Prediction($outcome);
    }

    /**
     * Compute the coefficients of the training data like ordinary least squares,
     * however add a regularization term to the equation.
     *
     * @param  array  $samples
     * @param  array  $outcomes
     * @return array
     */
    protected function computeCoefficients(array $samples, array $outcomes) : array
    {
        foreach ($samples as &$sample) {
            array_unshift($sample, 1);
        }

        $x = MatrixFactory::create($samples);
        $y = new Vector($outcomes);

        $identity = new DiagonalMatrix(array_replace([0], array_fill(1, $x->getN() - 1, 1)));

        return $x->transpose()->multiply($x)
            ->add($identity->scalarMultiply($this->alpha))
            ->inverse()
            ->multiply($x->transpose()->multiply($y))
            ->getColumn(0);
    }
}
