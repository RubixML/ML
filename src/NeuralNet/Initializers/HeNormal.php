<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers;

use NumPower;
use NDArray;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;

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
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        $stdDev = sqrt(2 / $fanOut);

        return NumPower::truncatedNormal(shape: [$fanOut, $fanIn], loc: 0.0, scale: $stdDev);
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
