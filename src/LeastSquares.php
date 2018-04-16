<?php

namespace Rubix\Engine;

use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\Engine\Connectors\Persistable;
use InvalidArgumentException;

class LeastSquares implements Regression, Persistable
{
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
     * Learn the coefficients of the training data.
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
     * Compute the coefficients of the training data by solving for the normal
     * equation. The resulting equation is the line that minimizes the sum of
     * the squares of the errors.
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

        return $samples->transpose()->multiply($samples)->inverse()
            ->multiply($samples->transpose()->multiply($outcomes))
            ->getColumn(0);
    }
}
