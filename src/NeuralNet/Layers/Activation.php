<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

/**
 * Activation
 *
 * Activation layers apply a user-defined non-linear activation function to their
 * inputs.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Activation implements Hidden
{
    /**
     * The function that computes the output of the layer.
     *
     * @var ActivationFunction
     */
    protected ActivationFunction $activationFn;

    /**
     * The width of the layer.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The memorized input matrix.
     *
     * @var NDArray|null
     */
    protected ?NDArray $input = null;

    /**
     * The memorized activation matrix.
     *
     * @var NDArray|null
     */
    protected ?NDArray $output = null;

    /**
     * @param ActivationFunction $activationFn
     */
    public function __construct(ActivationFunction $activationFn)
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->activationFn = $activationFn;
    }

    /**
     * Return the width of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return positive-int
     */
    public function width() : int
    {
        if ($this->width === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @internal
     *
     * @param positive-int $fanIn
     * @return positive-int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray
    {
        $output = $this->activationFn->activate($input);

        $this->input = $input;
        $this->output = $output;

        return $output;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        return $this->activationFn->activate($input);
    }

    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->input or !$this->output) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $input = $this->input;
        $output = $this->output;

        $this->input = $this->output = null;

        return new Deferred(
            [$this, 'gradient'],
            [$input, $output, $prevGradient]
        );
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @param NDArray $output
     * @param Deferred $prevGradient
     * @return NDArray
     */
    public function gradient(NDArray $input, NDArray $output, Deferred $prevGradient) : NDArray
    {
        return NumPower::multiply(
            $this->activationFn->differentiate($input, $output),
            $prevGradient()
        );
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
        return "Activation (activation fn: {$this->activationFn})";
    }
}
