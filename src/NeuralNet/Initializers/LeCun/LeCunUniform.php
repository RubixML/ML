<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\LeCun;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\Initializers\Base\AbstractInitializer;

/**
 * Le Cun Uniform
 *
 * Proposed by Yan Le Cun in a paper in 1998, this initializer was one of the
 * first published attempts to control the variance of activations between
 * layers through weight initialization. It remains a good default choice for
 * many hidden layer configurations. It draws from a uniform distribution
 * with limits +/- sqrt(3 / fanOut).
 *
 * References:
 * [1] Y. Le Cun et al. (1998). Efficient Backprop.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class LeCunUniform extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        $limit = sqrt(3 / $fanOut);

        return NumPower::uniform(size: [$fanOut, $fanIn], low: -$limit, high: $limit);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Le Cun Uniform';
    }
}
