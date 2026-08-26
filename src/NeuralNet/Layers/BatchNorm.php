<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameter;
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

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

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

        [$n, $m] = $input->shape();

        // Column-wise mean across samples (axis 1), length n
        $sum = NumPower::sum($input, axis: 1);
        $mean = NumPower::divide($sum, $m);

        // Center the input: broadcast mean to [n, m]
        $centered = NumPower::subtract($input, NumPower::reshape($mean, [$n, 1]));

        // Column-wise variance across samples (axis 1)
        $centeredSq = NumPower::multiply($centered, $centered);
        $varSum = NumPower::sum($centeredSq, axis: 1);
        $variance = NumPower::divide($varSum, $m);
        $variance = NumPower::clip($variance, EPSILON, PHP_FLOAT_MAX);

        // Inverse std from clipped variance
        $stdInv = NumPower::reciprocal(NumPower::sqrt($variance));

        // Normalize: (x - mean) * stdInv
        $xHat = NumPower::multiply($centered, NumPower::reshape($stdInv, [$n, 1]));

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

        $n = $input->shape()[0];

        $varianceClipped = NumPower::clip($this->variance, EPSILON, PHP_FLOAT_MAX);

        $xHat = NumPower::divide(
            NumPower::subtract($input, NumPower::reshape($this->mean, [$n, 1])),
            NumPower::reshape(NumPower::sqrt($varianceClipped), [$n, 1])
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

        // Sum across samples (axis 1) for parameter gradients
        $dBeta = NumPower::sum($dOut, axis: 1);
        $dGamma = NumPower::sum(NumPower::multiply($dOut, $this->xHat), axis: 1);
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
        $xHatSigma = NumPower::sum(NumPower::multiply($dXHat, $xHat), axis: 1);
        $dXHatSigma = NumPower::sum($dXHat, axis: 1);

        [$n, $m] = $dOut->shape();

        // Compute gradient per formula: dX = (dXHat * m - dXHatSigma - xHat * xHatSigma) * (stdInv / m)
        $dXHatTimesM = NumPower::multiply($dXHat, $m);

        $dXHatSigmaColumn = NumPower::reshape($dXHatSigma, [$n, 1]);
        $xHatSigmaColumn = NumPower::reshape($xHatSigma, [$n, 1]);
        $xHatTimesXHatSigma = NumPower::multiply($xHat, $xHatSigmaColumn);

        $numerator = NumPower::subtract(
            NumPower::subtract($dXHatTimesM, $dXHatSigmaColumn),
            $xHatTimesXHatSigma
        );

        $stdInvOverMColumn = NumPower::reshape(NumPower::divide($stdInv, $m), [$n, 1]);

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
