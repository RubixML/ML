<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions\ReLU;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU\ReLU;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class ReLUBench
{
    /**
     * @var NDArray
     */
    protected NDArray $z;

    /**
     * @var NDArray
     */
    protected NDArray $computed;

    /**
     * @var ReLU
     */
    protected ReLU $activationFn;

    public function setUp() : void
    {
        $this->z = NumPower::uniform(size: [500, 500], low: -1.0, high: 1.0);

        $this->computed = NumPower::uniform(size: [500, 500], low: -1.0, high: 1.0);

        $this->activationFn = new ReLU();
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
        $this->activationFn->differentiate($this->z);
    }
}
