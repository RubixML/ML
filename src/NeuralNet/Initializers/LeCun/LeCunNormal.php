<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\LeCun;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\Initializers\Base\AbstractInitializer;

/**
 * Le Cun Normal
 *
 * Proposed by Yan Le Cun in a paper in 1998, this initializer was one of the
 * first published attempts to control the variance of activations between
 * layers through weight initialization. It remains a good default choice for
 * many hidden layer configurations. It draws from a truncated
 * normal distribution with mean 0 and standard deviation sqrt(1 / fanOut).
 *
 * References:
 * [1] Y. Le Cun et al. (1998). Efficient Backprop.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class LeCunNormal extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        $stdDev = sqrt(1 / $fanOut);

        return NumPower::truncatedNormal(size: [$fanOut, $fanIn], scale: $stdDev);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Le Cun Normal';
    }
}
