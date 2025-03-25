<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\He;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\Initializers\Base\Contracts\AbstractInitializer;

/**
 * He Normal
 *
 * The He initializer was designed for hidden layers that feed into rectified
 * linear layers such ReLU, Leaky ReLU, ELU, and SELU. It draws from a truncated
 * normal distribution with mean 0 and standart deviation sqrt(2 / fanOut).
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class HeNormal extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateInitParams(fanIn: $fanIn, fanOut: $fanOut);

        $std = sqrt(2 / $fanOut);

        return NumPower::truncatedNormal(size: [$fanOut, $fanIn], loc: 0.0, scale: $std);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'He Normal';
    }
}
