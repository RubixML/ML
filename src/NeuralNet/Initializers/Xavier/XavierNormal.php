<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Xavier;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\Initializers\Base\Contracts\AbstractInitializer;

/**
 * Xavier Normal
 *
 * The Xavier 1 initializer draws from a truncated normal distribution with
 * mean 0 and standard deviation squal sqrt(2 / (fanIn + fanOut)). This initializer is
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
class XavierNormal extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateInitParams(fanIn: $fanIn, fanOut: $fanOut);

        $std = sqrt(2 / ($fanOut + $fanIn));

        return NumPower::truncatedNormal(size: [$fanOut, $fanIn], loc: 0.0, scale: $std);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Xavier Normal';
    }
}
