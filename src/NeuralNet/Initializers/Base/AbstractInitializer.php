<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Base;

use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

/**
 * Abstract Initializer for init params validation
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
abstract class AbstractInitializer implements Initializer
{
    /**
     * Validating initializer parameters
     *
     * @param int $fanIn The number of input connections per neuron
     * @param int $fanOut The number of output connections per neuron
     * @throws InvalidFanInException Initializer parameter fanIn is less than 1
     * @throws InvalidFanOutException Initializer parameter fanOut is less than 1
     */
    protected function validateFanInFanOut(int $fanIn, int $fanOut) : void
    {
        if ($fanIn < 1) {
            throw new InvalidFanInException(message: "Fan in cannot be less than 1, $fanIn given");
        }

        if ($fanOut < 1) {
            throw new InvalidFanOutException(message: "Fan out cannot be less than 1, $fanOut given");
        }
    }
}
