<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

/**
 * Placeholder 1D
 *
 * The Placeholder 1D input layer represents the *future* input values of a mini
 * batch (matrix) of single dimensional tensors (vectors) to the neural network.
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
     * @throws \InvalidArgumentException
     */
    public function __construct(int $inputs)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of input nodes must'
            . " be greater than 0, $inputs given.");
        }

        $this->inputs = $inputs;
    }

    /**
     * @return int|null
     */
    public function width() : ?int
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
     * @param \Rubix\Tensor\Matrix $input
     * @throws \InvalidArgumentException
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($input->m() !== $this->inputs) {
            throw new InvalidArgumentException('The number of feature columns'
                . ' must equal the number of inputs. '
                . " {$input->m()} found, but $this->inputs needed.");
        }

        return $input;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->forward($input);
    }
}
