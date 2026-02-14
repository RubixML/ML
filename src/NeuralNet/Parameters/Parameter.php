<?php

namespace Rubix\ML\NeuralNet\Parameters;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;

/**
 * Parameter
 *
 * A wrapper over an NDArray from NumPower that marks the parameter as trainable
 * and provides updates via the optimizer.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */

/**
 * Parameter
 */
class Parameter
{
    /**
     * The auto incrementing id.
     *
     * @var int
     */
    protected static int $counter = 0;

    /**
     * The unique identifier of the parameter.
     *
     * @var int
     */
    protected int $id;

    /**
     * The parameter.
     *
     * @var NDArray
     */
    protected NDArray $param;

    /**
     * @param NDArray $param
     */
    public function __construct(NDArray $param)
    {
        $this->id = self::$counter++;
        $this->param = $param;
    }

    /**
     * Return the unique identifier of the parameter.
     *
     * @return int
     */
    public function id() : int
    {
        return $this->id;
    }

    /**
     * Return the wrapped parameter.
     *
     * @return NDArray
     */
    public function param() : NDArray
    {
        return $this->param;
    }

    /**
     * Update the parameter with the gradient and optimizer.
     *
     * @param NDArray $gradient
     * @param Optimizer $optimizer
     */
    public function update(NDArray $gradient, Optimizer $optimizer) : void
    {
        $step = $optimizer->step($this, $gradient);

        $this->param = NumPower::subtract($this->param, $step);
    }

    /**
     * Perform a deep copy of the object upon cloning.
     *
     * Cloning an NDArray directly may trigger native memory corruption in some
     * NumPower builds (e.g. heap corruption/segfaults when parameters are
     * snapshotted during training). To make cloning deterministic and stable we
     * deep-copy through a PHP array roundtrip: NDArray -> PHP array -> NDArray.
     */
    public function __clone() : void
    {
        $this->param = NumPower::array($this->param->toArray());
    }
}
