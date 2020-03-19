<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class ThresholdedReLUBench
{
    /**
     * @var \Tensor\Matrix
     */
    protected $z;

    /**
     * @var \Tensor\Matrix
     */
    protected $computed;

    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU
     */
    protected $activationFn;

    public function setUp() : void
    {
        $this->z = Matrix::uniform(500, 500);

        $this->computed = Matrix::uniform(500, 500);

        $this->activationFn = new ThresholdedReLU();
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function compute() : void
    {
        $this->activationFn->compute($this->z);
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function differentiate() : void
    {
        $this->activationFn->differentiate($this->z, $this->computed);
    }
}
