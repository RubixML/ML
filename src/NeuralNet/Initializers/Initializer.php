<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers;

use NDArray;
use Stringable;
use Rubix\ML\Exceptions\InvalidFanInException;
use Rubix\ML\Exceptions\InvalidFanOutException;

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
     * @param int<1, max> $fanIn
     * @param int<1, max> $fanOut
     * @param string $dataType
     * @throws InvalidFanInException
     * @throws InvalidFanOutException
     * @return NDArray
     */
    public function initialize(int $fanIn, int $fanOut, string $dataType) : NDArray;
}
