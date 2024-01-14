<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class LeakyReLUBench
{
    /**
     * @var Matrix
     */
    protected $z;

    /**
     * @var Matrix
     */
    protected $computed;

    /**
     * @var LeakyReLU
     */
    protected $activationFn;

    public function setUp() : void
    {
        $this->z = Matrix::uniform(500, 500);

        $this->computed = Matrix::uniform(500, 500);

        $this->activationFn = new LeakyReLU();
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function activate() : void
    {
        $this->activationFn->activate($this->z);
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
