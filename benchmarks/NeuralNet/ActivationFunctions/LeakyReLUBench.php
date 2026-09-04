<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class LeakyReLUBench
{
    /**
     * @var NDArray
     */
    protected Matrix $z;

    /**
     * @var NDArray
     */
    protected Matrix $computed;

    /**
     * @var LeakyReLU
     */
    protected LeakyReLU $activationFn;

    public function setUp() : void
    {
        $this->z = NumPower::uniform([500, 500]);

        $this->computed = NumPower::uniform([500, 500]);

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
