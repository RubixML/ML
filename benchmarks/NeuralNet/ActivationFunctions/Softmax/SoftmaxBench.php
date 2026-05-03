<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions\Softmax;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax\Softmax;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class SoftmaxBench
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
     * @var Softmax
     */
    protected Softmax $activationFn;

    public function setUp() : void
    {
        $this->z = NumPower::uniform(size: [100, 100], low: -1.0, high: 1.0);

        $this->computed = NumPower::uniform(size: [100, 100], low: -1.0, high: 1.0);

        $this->activationFn = new Softmax();
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
        $this->activationFn->differentiate($this->computed);
    }
}
