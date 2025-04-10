<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Uniform;

use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Initializers\Base\AbstractInitializer;
use Rubix\ML\NeuralNet\Initializers\Uniform\Exceptions\InvalidBetaException;

/**
 * Uniform
 *
 * Generates a random uniform distribution centered at 0 and bounded at
 * both ends by the parameter beta.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class Uniform extends AbstractInitializer
{
    /**
     * @param float $beta The upper and lower bound of the distribution.
     * @throws InvalidArgumentException
     */
    public function __construct(protected float $beta = 0.5)
    {
        if ($this->beta <= 0.0) {
            throw new InvalidBetaException(
                message: "Beta cannot be less than or equal to 0, $beta given."
            );
        }
    }

    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        return NumPower::uniform(
            size: [$fanOut, $fanIn],
            low: -$this->beta,
            high: $this->beta
        );
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return "Uniform (beta: {$this->beta})";
    }
}
