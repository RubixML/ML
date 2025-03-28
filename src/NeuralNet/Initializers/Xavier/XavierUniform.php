<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Xavier;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\Initializers\Base\Contracts\AbstractInitializer;

/**
 * Xavier Uniform
 *
 * The Xavier 1 initializer draws from a uniform distribution [-limit, limit]
 * where *limit* is squal to sqrt(6 / (fanIn + fanOut)). This initializer is
 * best suited for layers that feed into an activation layer that outputs a
 * value between 0 and 1 such as Softmax or Sigmoid.
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
class XavierUniform extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateInitParams(fanIn: $fanIn, fanOut: $fanOut);

        $limit = sqrt(6 / ($fanOut + $fanIn));

        return NumPower::uniform(size: [$fanOut, $fanIn], low: -$limit, high: $limit);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Xavier Uniform';
    }
}
