<?php

namespace Rubix\ML\Benchmarks\NeuralNet\Optimizers;

use Tensor\Matrix;
use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Clipped;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

/**
 * @Groups({"Optimizers"})
 * @BeforeMethods({"setUp"})
 */
class ClippedBench
{
    /**
     * @var list<Parameter>
     */
    protected array $params;

    /**
     * @var list<Tensor>
     */
    protected array $gradients;

    /**
     * @var Clipped
     */
    protected Clipped $optimizer;

    public function setUp() : void
    {
        $this->params = [];
        $this->gradients = [];

        for ($i = 0; $i < 10; ++$i) {
            $this->params[] = new Parameter(Matrix::uniform(64, 64));
            $this->gradients[] = Matrix::uniform(64, 64);
        }

        $this->optimizer = new Clipped(new Stochastic(new Constant(0.01)), 1.0);
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function step() : void
    {
        $gradients = [];

        foreach ($this->params as $i => $param) {
            $gradients[] = [$param, $this->gradients[$i]];
        }

        $this->optimizer->step($gradients);
    }
}
