<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Output;

interface Network
{
    /**
     * Return all the layers in the network.
     *
     * @return array
     */
    public function layers() : array;

    /**
     * Return the input layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Input
     */
    public function input() : Input;

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return array
     */
    public function hidden() : array;

    /**
     * Return the output layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Output
     */
    public function output() : Output;

    /**
     * Return the parametric layers of the network.
     *
     * @return array
     */
    public function parametric() : array;

    /**
     * The depth of the network. i.e. the number of parametric layers.
     *
     * @return int
     */
    public function depth() : int;
}
