<?php

namespace Rubix\ML\Benchmarks\NeuralNet\ActivationFunctions\HyperbolicTangent;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent\HyperbolicTangent;

/**
 * @Groups({"ActivationFunctions"})
 * @BeforeMethods({"setUp"})
 */
class HyperbolicTangentBench
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
     * @var HyperbolicTangent
     */
    protected HyperbolicTangent $activationFn;

    public function setUp() : void
    {
        $this->z = NumPower::uniform([500, 500], low: -1.0, high: 1.0);

        $this->computed = NumPower::uniform([500, 500], low: -1.0, high: 1.0);

        $this->activationFn = new HyperbolicTangent();
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
