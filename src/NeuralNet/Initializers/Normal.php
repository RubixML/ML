<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers;

use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Traits\AssertsShapes;
use Rubix\ML\Exceptions\InvalidStandardDeviationException;

/**
 * Normal
 *
 * Generates a random weight matrix from a Gaussian distribution with user-specified standard
 * deviation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class Normal implements Initializer
{
    use AssertsShapes;

    /**
     * @param float $stdDev
     * @throws InvalidArgumentException
     */
    public function __construct(protected float $stdDev = 0.05)
    {
        if ($this->stdDev <= 0.0) {
            throw new InvalidStandardDeviationException(
                message: "Standard deviation must be greater than 0, $stdDev given."
            );
        }

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
    }

    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut, string $dataType) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        return NumPower::normal(
            [$fanOut, $fanIn],
            loc: 0.0,
            scale: $this->stdDev,
            dtype: $dataType
        );
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Normal (stdDev: {$this->stdDev})";
    }
}
