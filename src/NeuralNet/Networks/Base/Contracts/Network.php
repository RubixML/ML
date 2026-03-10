<?php

namespace Rubix\ML\NeuralNet\Networks\Base\Contracts;

use Rubix\ML\NeuralNet\Layers\Base\Contracts\Hidden;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Input;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Output;
use Traversable;

/**
 * Network
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Network
{
    /**
     * Return the layers of the network.
     *
     * @return Traversable
     */
    public function layers() : Traversable;

    /**
     * Return the input layer.
     *
     * @return Input
     */
    public function input() : Input;

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return list<Hidden>
     */
    public function hidden() : array;

    /**
     * Return the output layer.
     *
     * @return Output
     */
    public function output() : Output;
}
