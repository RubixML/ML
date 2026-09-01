<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers;

use NumPower;
use NDArray;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Traits\AssertsShapes;

/**
 * Xavier 1 Normal
 *
 * The Xavier 1 Normal initializer draws from a truncated normal distribution with
 * mean 0 and standard deviation equal to sqrt(2 / (fanIn + fanOut)). This
 * initializer is best suited for layers that feed into an activation layer that
 * outputs a value between 0 and 1 such as Softmax or Sigmoid.
 *
 * References:
 * [1] X. Glorot et al. (2010). Understanding the Difficulty of Training Deep
 * Feedforward Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class Xavier1Normal implements Initializer
{
    use AssertsShapes;

    public function __construct()
    {
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

        $stdDev = sqrt(2 / ($fanOut + $fanIn));

        return NumPower::truncatedNormal(
            [$fanOut, $fanIn],
            loc: 0.0,
            scale: $stdDev,
            dtype: $dataType
        );
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Xavier-1 Normal';
    }
}
