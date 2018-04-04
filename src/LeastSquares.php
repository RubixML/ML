<?php

namespace Rubix\Engine;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\MatrixFactory;

class LeastSquares implements Regression
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
     * @return void
     */
    public function __construct()
    {
        //
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
     * Learn the coefficients of the training data.
     *
     * @param  array  $samples
     * @param  array  $outcomes
     * @return void
     */
    public function train(array $samples, array $outcomes) : void
    {
        foreach ($samples as &$sample) {
            array_unshift($sample, 1);
        }

        $samples = MatrixFactory::create($samples);
        $outcomes = MatrixFactory::create([new Vector($outcomes)]);

        $coefficients = $this->computeCoefficients($samples, $outcomes)->getColumn(0);

        $this->intercept = array_shift($coefficients);
        $this->coefficients = $coefficients;
    }

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        $outcome = $this->intercept;

        foreach ($this->coefficients as $i => $coefficient) {
            $outcome += $coefficient * $sample[$i];
        }

        return [
            'outcome' => $outcome,
        ];
    }

    /**
     * Compute the coefficients of the training data by solving the normal equation.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $samples
     * @param  \MathPHP\LinearAlgebra\Matrix  $outcomes
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    protected function computeCoefficients(Matrix $samples, Matrix $outcomes) : Matrix
    {
        return $samples->transpose()->multiply($samples)->inverse()
            ->multiply($samples->transpose()->multiply($outcomes));
    }
}
