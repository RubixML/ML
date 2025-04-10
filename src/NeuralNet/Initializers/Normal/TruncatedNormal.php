<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Normal;

use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Initializers\Base\AbstractInitializer;
use Rubix\ML\NeuralNet\Initializers\Normal\Exceptions\InvalidStandardDeviationException;

/**
 * Truncated Normal
 *
 * The values generated are similar to values from a Normal initializer,
 * except that values more than two standard deviations from the mean
 * are discarded and re-drawn.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class TruncatedNormal extends AbstractInitializer
{
    /**
     * @param float $stdDev The standard deviation of the distribution to sample from
     * @throws InvalidArgumentException
     */
    public function __construct(protected float $stdDev = 0.05)
    {
        if ($this->stdDev <= 0.0) {
            throw new InvalidStandardDeviationException(
                message: "Standard deviation must be greater than 0, $stdDev given."
            );
        }
    }

    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        return NumPower::truncatedNormal(size: [$fanOut, $fanIn], loc: 0.0, scale: $this->stdDev);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return "Truncated Normal (stdDev: {$this->stdDev})";
    }
}
