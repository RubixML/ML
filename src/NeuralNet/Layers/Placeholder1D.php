<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Placeholder 1D
 *
 * The Placeholder 1D input layer represents the *future* input values of a mini
 * batch (matrix) of single dimensional tensors (vectors) to the neural network.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Placeholder1D implements Input
{
    /**
     * The number of input nodes. i.e. feature inputs.
     *
     * @var int
     */
    protected $inputs;

    /**
     * @param int $inputs
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $inputs)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('Number of input nodes'
            . " must be greater than 0, $inputs given.");
        }

        $this->inputs = $inputs;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->inputs;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        return $this->inputs;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($input->m() !== $this->inputs) {
            throw new InvalidArgumentException('The number of features'
                . ' and input nodes must be equal,'
                . " $this->inputs expected but {$input->m()} given.");
        }

        return $input;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->forward($input);
    }
}
