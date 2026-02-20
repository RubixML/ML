<?php

namespace Rubix\ML\NeuralNet\Layers\BatchNorm;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Hidden;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Base\Initializer;
use Rubix\ML\NeuralNet\Initializers\Constant\Constant;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Generator;

use const Rubix\ML\EPSILON;

/**
 * Batch Norm
 *
 * Normalize the activations of the previous layer such that the mean activation
 * is close to 0 and the standard deviation is close to 1. Batch Norm can reduce
 * the amount of covariate shift within the network which makes it possible to use
 * higher learning rates and converge faster under some circumstances.
 *
 * References:
 * [1] S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network
 * Training by Reducing Internal Covariate Shift.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class BatchNorm implements Hidden, Parametric
{
    /**
     * The decay rate of the previous running averages of the global mean and variance.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The initializer for the beta parameter.
     *
     * @var Initializer
     */
    protected Initializer $betaInitializer;

    /**
     * The initializer for the gamma parameter.
     *
     * @var Initializer
     */
    protected Initializer $gammaInitializer;

    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The learnable centering parameter.
     *
     * @var Parameter|null
     */
    protected ?Parameter $beta = null;

    /**
     * The learnable scaling parameter.
     *
     * @var Parameter|null
     */
    protected ?Parameter $gamma = null;

    /**
     * The running mean of each input dimension.
     *
     * @var NDArray|null
     */
    protected ?NDArray $mean = null;

    /**
     * The running variance of each input dimension.
     *
     * @var NDArray|null
     */
    protected ?NDArray $variance = null;

    /**
     * A cache of inverse standard deviations calculated during the forward pass.
     *
     * @var NDArray|null
     */
    protected ?NDArray $stdInv = null;

    /**
     * A cache of normalized inputs to the layer.
     *
     * @var NDArray|null
     */
    protected ?NDArray $xHat = null;

    /**
     * Row-wise or column-wise normalization.
     *
     * @var int
     */
    protected const int AXIS_SAMPLES = 0;
    protected const int AXIS_FEATURES = 1;

    /**
     * @param float $decay
     * @param Initializer|null $betaInitializer
     * @param Initializer|null $gammaInitializer
     * @throws InvalidArgumentException
     */
    public function __construct(float $decay = 0.1, ?Initializer $betaInitializer = null, ?Initializer $gammaInitializer = null)
    {
        if ($decay < 0.0 or $decay > 1.0) {
            throw new InvalidArgumentException("Decay must be between 0 and 1, $decay given.");
        }

        $this->decay = $decay;
        $this->betaInitializer = $betaInitializer ?? new Constant(0.0);
        $this->gammaInitializer = $gammaInitializer ?? new Constant(1.0);
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

        // Initialize beta and gamma as vectors of length fanOut
        // We request a [fanOut, 1] NDArray and then flatten to 1-D
        $betaMat = $this->betaInitializer->initialize(1, $fanOut);
        $gammaMat = $this->gammaInitializer->initialize(1, $fanOut);

        $beta = NumPower::flatten($betaMat);
        $gamma = NumPower::flatten($gammaMat);

        $this->beta = new Parameter($beta);
        $this->gamma = new Parameter($gamma);

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @throws RuntimeException
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        [$m, $n] = $input->shape();

        // Row-wise mean across features (axis 1), length m
        $sum = NumPower::sum($input, axis: self::AXIS_FEATURES);
        $mean = NumPower::divide($sum, $n);

        // Center the input: broadcast mean to [m, n]
        $centered = NumPower::subtract($input, NumPower::reshape($mean, [$m, 1]));

        // Row-wise variance across features (axis 1)
        $centeredSq = NumPower::multiply($centered, $centered);
        $varSum = NumPower::sum($centeredSq, axis: self::AXIS_FEATURES);
        $variance = NumPower::divide($varSum, $n);
        $variance = NumPower::clip($variance, EPSILON, PHP_FLOAT_MAX);

        // Inverse std from clipped variance
        $stdInv = NumPower::reciprocal(NumPower::sqrt($variance));

        // Normalize: (x - mean) * stdInv
        $xHat = NumPower::multiply($centered, NumPower::reshape($stdInv, [$m, 1]));

        // Initialize running stats if needed
        if (!$this->mean or !$this->variance) {
            $this->mean = $mean;
            $this->variance = $variance;
        }

        // Update running mean/variance using exponential moving average (EMA)
        // Convention: running = running*(1 - decay) + current*decay
        $this->mean = NumPower::add(
            NumPower::multiply($this->mean, 1.0 - $this->decay),
            NumPower::multiply($mean, $this->decay)
        );

        $this->variance = NumPower::add(
            NumPower::multiply($this->variance, 1.0 - $this->decay),
            NumPower::multiply($variance, $this->decay)
        );

        $this->stdInv = $stdInv;
        $this->xHat = $xHat;

        // gamma * xHat + beta (per-column scale/shift) using NDArray ops
        return NumPower::add(NumPower::multiply($xHat, $this->gamma->param()), $this->beta->param());
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @throws RuntimeException
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        if (!$this->mean or !$this->variance or !$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $m = $input->shape()[0];

        // Use clipped variance for numerical stability during inference
        $varianceClipped = NumPower::clip($this->variance, EPSILON, PHP_FLOAT_MAX);
        $xHat = NumPower::divide(
            NumPower::subtract($input, NumPower::reshape($this->mean, [$m, 1])),
            NumPower::reshape(NumPower::sqrt($varianceClipped), [$m, 1])
        );

        return NumPower::add(
            NumPower::multiply(
                $xHat,
                $this->gamma->param()
            ),
            $this->beta->param()
        );
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
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
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->stdInv or !$this->xHat) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $dOut = $prevGradient();
        // Sum across samples (axis 0) for parameter gradients
        $dBeta = NumPower::sum($dOut, axis: self::AXIS_SAMPLES);
        $dGamma = NumPower::sum(NumPower::multiply($dOut, $this->xHat), axis: self::AXIS_SAMPLES);
        $gamma = $this->gamma->param();

        $this->beta->update($dBeta, $optimizer);
        $this->gamma->update($dGamma, $optimizer);

        $stdInv = $this->stdInv;
        $xHat = $this->xHat;

        $this->stdInv = $this->xHat = null;

        return new Deferred(
            [$this, 'gradient'],
            [$dOut, $gamma, $stdInv, $xHat]
        );
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param NDArray $dOut
     * @param NDArray $gamma
     * @param NDArray $stdInv
     * @param NDArray $xHat
     * @return NDArray
     */
    public function gradient(NDArray $dOut, NDArray $gamma, NDArray $stdInv, NDArray $xHat) : NDArray
    {
        $dXHat = NumPower::multiply($dOut, $gamma);
        $xHatSigma = NumPower::sum(NumPower::multiply($dXHat, $xHat), axis: self::AXIS_FEATURES);
        $dXHatSigma = NumPower::sum($dXHat, axis: self::AXIS_FEATURES);

        $m = $dOut->shape()[0];

        // Compute gradient per formula: dX = (dXHat * m - dXHatSigma - xHat * xHatSigma) * (stdInv / m)
        $dXHatTimesM = NumPower::multiply($dXHat, $m);
        $dXHatSigmaColumn = NumPower::reshape($dXHatSigma, [$m, 1]);
        $xHatSigmaColumn = NumPower::reshape($xHatSigma, [$m, 1]);
        $xHatTimesXHatSigma = NumPower::multiply($xHat, $xHatSigmaColumn);

        $numerator = NumPower::subtract(
            NumPower::subtract($dXHatTimesM, $dXHatSigmaColumn),
            $xHatTimesXHatSigma
        );

        $stdInvOverMColumn = NumPower::reshape(NumPower::divide($stdInv, $m), [$m, 1]);

        return NumPower::multiply($numerator, $stdInvOverMColumn);

    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'beta' => $this->beta;
        yield 'gamma' => $this->gamma;
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @internal
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
        $this->gamma = $parameters['gamma'];
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
        return "Batch Norm (decay: {$this->decay}, beta initializer: {$this->betaInitializer},"
            . " gamma initializer: {$this->gammaInitializer})";
    }
}
