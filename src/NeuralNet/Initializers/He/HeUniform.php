<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\He;

use NumPower;
use Rubix\ML\NeuralNet\Initializers\Base\Contracts\AbstractInitializer;

/**
 * He Uniform
 *
 * The He initializer was designed for hidden layers that feed into rectified
 * linear layers such ReLU, Leaky ReLU, ELU, and SELU. It draws from a uniform
 * distribution with limits +/- sqrt(6 / fanOut).
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
class HeUniform extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NumPower
    {
        $this->validateInitParams(fanIn: $fanIn, fanOut: $fanOut);

        $limit = sqrt(6 / $fanOut);

        return NumPower::uniform(size: [$fanOut, $fanIn], low: -$limit, high: $limit);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'He Uniform';
    }
}
