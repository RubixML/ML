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
 * Le Cun Normal
 *
 * Proposed by Yan Le Cun in a paper in 1998, this initializer was one of the
 * first published attempts to control the variance of activations between
 * layers through weight initialization. It remains a good default choice for
 * many hidden layer configurations. It draws from a truncated
 * normal distribution with mean 0 and standard deviation sqrt(1 / fanIn).
 *
 * References:
 * [1] Y. Le Cun et al. (1998). Efficient Backprop.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class LeCunNormal implements Initializer
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

        $stdDev = sqrt(1 / $fanIn);

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
     * @return string
     */
    public function __toString() : string
    {
        return 'Le Cun Normal';
    }
}
