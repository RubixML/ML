<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers\Base\Contracts;

use NDArray;
use Stringable;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanInException;
use Rubix\ML\NeuralNet\Initializers\Base\Exceptions\InvalidFanOutException;

/**
 * Initializer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
interface Initializer extends Stringable
{
    /**
     * Initialize a weight matrix W in the dimensions `fanIn` x `fanOut`.
     *
     * @param int<1, max> $fanIn The number of input connections per neuron
     * @param int<1, max> $fanOut The number of output connections per neuron
     * @throws InvalidFanInException Initializer parameter `fanIn` is less than 1
     * @throws InvalidFanOutException Initializer parameter `fanOut` is less than 1
     * @return NDArray The initialized weight matrix of shape [fanOut, fanIn]
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray;
}
