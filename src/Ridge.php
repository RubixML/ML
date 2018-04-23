<?php

namespace Rubix\Engine;

use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;

class Ridge implements Regression, Persistable
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
    public function __construct(float $alpha = 0.5)
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
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (!$data instanceof SupervisedDataset) {
            throw new InvalidArgumentException('This estimator requires a supervised dataset.');
        }

        if (in_array(self::CATEGORICAL, $data->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $coefficients = $this->computeCoefficients($data->samples(), $data->outcomes());

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

        foreach ($this->coefficients as $i => $coefficient) {
            $outcome += $coefficient * $sample[$i];
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

        $samples = MatrixFactory::create($samples);
        $outcomes = MatrixFactory::create([new Vector($outcomes)]);

        return $samples->transpose()->multiply($samples)
            ->add(MatrixFactory::identity($samples->getN())->scalarMultiply($this->alpha))
            ->inverse()->multiply($samples->transpose()->multiply($outcomes))
            ->getColumn(0);
    }
}
