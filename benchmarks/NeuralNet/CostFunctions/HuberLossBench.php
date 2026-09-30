<?php

namespace Rubix\ML\Benchmarks\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

/**
 * @Groups({"CostFunctions"})
 * @BeforeMethods({"setUp"})
 */
class HuberLossBench
{
    /**
     * @var Matrix
     */
    protected Matrix $z;

    /**
     * @var Matrix
     */
    protected Matrix $y;

    /**
     * @var HuberLoss
     */
    protected HuberLoss $lossFn;

    public function setUp() : void
    {
        $this->z = Matrix::uniform(500, 500);

        $this->y = Matrix::uniform(500, 500);

        $this->lossFn = new HuberLoss();
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function compute() : void
    {
        $this->lossFn->compute($this->z, $this->y);
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function differentiate() : void
    {
        $this->lossFn->differentiate($this->z, $this->y);
    }
}
