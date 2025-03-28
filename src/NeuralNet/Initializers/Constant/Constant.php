<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Constant;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\Initializers\Base\Contracts\AbstractInitializer;

/**
 * Constant
 *
 * Initialize the parameter to a user specified constant value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class Constant extends AbstractInitializer
{
    /**
     * @param float $value The value to initialize the parameter to
     */
    public function __construct(protected float $value = 0.0)
    {
    }

    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateInitParams(fanIn: $fanIn, fanOut: $fanOut);

        return NumPower::full(
            shape: [$fanOut, $fanIn],
            fill_value: $this->value
        );
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return "Constant (value: {$this->value})";
    }
}
