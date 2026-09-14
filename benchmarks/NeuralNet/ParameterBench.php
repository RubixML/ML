<?php

namespace Rubix\ML\Benchmarks\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;

/**
 * @Groups({"NeuralNet"})
 * @BeforeMethods({"setUp"})
 */
class ParameterBench
{
    /**
     * @var Parameter
     */
    protected Parameter $param;

    public function setUp() : void
    {
        $this->param = new Parameter(Matrix::uniform(128, 1024));

        $this->param->accumulateGradient(Matrix::uniform(128, 1024));
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function gradientNorm() : float
    {
        return $this->param->gradientNorm();
    }
}
