<?php

namespace Rubix\ML\NeuralNet\Layers\Dropout;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Hidden;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Dropout
 *
 * Dropout is a regularization technique for reducing overfitting in neural
 * networks by preventing complex co-adaptations on training data. It works
 * by temporarily disabling neurons during each training pass. It also is a
 * very efficient way of performing model averaging with neural networks.
 *
 * References:
 * [1] N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural
 * Networks from Overfitting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dropout implements Hidden
{
    /**
     * The ratio of neurons that are dropped during each training pass.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    protected float $scale;

    /**
     * The width of the layer.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The memoized dropout mask.
     *
     * @var NDArray|null
     */
    protected ?NDArray $mask = null;

    /**
     * @param float $ratio
     * @throws InvalidArgumentException
     */
    public function __construct(float $ratio = 0.5)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException("Ratio must be between 0 and 1, $ratio given.");
        }

        $this->ratio = $ratio;
        $this->scale = 1.0 / (1.0 - $ratio);
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
     * @param NDArray|null $mask Custom dropout mask to use instead of generating one.
     * @return NDArray
     */
    public function forward(NDArray $input, ?NDArray $mask = null) : NDArray
    {
        if ($mask === null) {
            // Build dropout mask using PHP's RNG. Each unit is kept with
            // probability (1 - ratio) and scaled by $this->scale.
            $inputArray = $input->toArray();

            $maskArray = [];

            foreach ($inputArray as $i => $row) {
                foreach ($row as $j => $_value) {
                    $u = rand() / getrandmax();

                    $maskArray[$i][$j] = $u > $this->ratio ? $this->scale : 0.0;
                }
            }

            $mask = NumPower::array($maskArray);
        }

        $output = NumPower::multiply($input, $mask);

        $this->mask = $mask;

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
        return $input;
    }

    /**
     * Calculate the gradients of the layer and update the parameters.
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
        if (!$this->mask) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $mask = $this->mask;

        $this->mask = null;

        return new Deferred([$this, 'gradient'], [$prevGradient, $mask]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @param NDArray $mask
     * @return NDArray
     */
    public function gradient(Deferred $prevGradient, NDArray $mask) : NDArray
    {
        /** @var NDArray $dOut */
        $dOut = $prevGradient();

        return NumPower::multiply($dOut, $mask);
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
        return "Dropout (ratio: {$this->ratio})";
    }
}
