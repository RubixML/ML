<?php

namespace Rubix\ML\NeuralNet\Layers\Placeholder1D;

use NDArray;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Input;
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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Placeholder1D implements Input
{
    /**
     * The number of input nodes. i.e. feature inputs.
     *
     * @var positive-int
     */
    protected int $inputs;

    /**
     * @param int $inputs
     * @throws InvalidArgumentException
     */
    public function __construct(int $inputs)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException("Number of input nodes must be greater than 0, $inputs given.");
        }

        $this->inputs = $inputs;
    }

    /**
     * @return positive-int
     */
    public function width() : int
    {
        return $this->inputs;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param positive-int $fanIn
     * @return positive-int
     */
    public function initialize(int $fanIn) : int
    {
        return $this->inputs;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param NDArray $input
     * @throws InvalidArgumentException
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray
    {
        $shape = $input->shape();

        if (empty($shape) || $shape[0] !== $this->inputs) {
            $features = $shape[0] ?? 0;

            throw new InvalidArgumentException(
                'The number of features and input nodes must be equal,'
                . " {$this->inputs} expected but {$features} given."
            );
        }

        return $input;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        return $this->forward($input);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Placeholder 1D (inputs: {$this->inputs})";
    }
}
