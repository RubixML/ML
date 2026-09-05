<?php

namespace Rubix\ML\Benchmarks\Functions;

use Tensor\Vector;

use function Rubix\ML\minmax;
use function min;
use function max;

/**
 * @Groups({"Functions"})
 * @BeforeMethods({"setUp"})
 */
class MinMaxBench
{
    protected const SIZE = 10000;

    /**
     * @var float[]
     */
    protected $values;

    public function setUp() : void
    {
        $this->values = Vector::rand(self::SIZE)->asArray();
    }

    /**
     * Compute the minimum and maximum in a single pass.
     *
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     * @return array{int|float,int|float}
     */
    public function minmax() : array
    {
        return minmax($this->values);
    }

    /**
     * Compute the minimum and maximum in two separate passes.
     *
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     * @return array{int|float,int|float}
     */
    public function twoPass() : array
    {
        return [
            min($this->values),
            max($this->values),
        ];
    }
}
