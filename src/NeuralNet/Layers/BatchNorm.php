<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Batch Norm
 *
 * Normalize the activations of the previous layer such that the mean activation
 * is close to 0 and the activation standard deviation is close to 1. Batch Norm
 * can be used to reduce the amount of covariate shift within the network
 * making it possible to use higher learning rates and converge faster under
 * some circumstances.
 *
 * References:
 * [1] S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network
 * Training by Reducing Internal Covariate Shift.
 * [2] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BatchNorm implements Hidden, Parametric
{
    /**
     * The variance smoothing parameter i.e a small value added to the variance
     * for numerical stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The learnable centering parameter.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $beta;

    /**
     * The learnable scaling parameter.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $gamma;

    /**
     * The running mean of each input dimension.
     *
     * @var array|null
     */
    protected $means;

    /**
     * The running variance of each input dimension.
     *
     * @var array|null
     */
    protected $variances;

    /**
     * The number of training samples that have passed through the layer so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * A cache of inverse standard deviations calculated during the forward pass.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $stdInv;

    /**
     * A cache of normalized inputs to the layer.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $xHat;

    /**
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $epsilon = 1e-8)
    {
        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Variance smoothing parameter'
                . 'must be greater than 0');
        }

        $this->epsilon = $epsilon;
    }

    /**
     * Return the width of the layer.
     * 
     * @return int|null
     */
    public function width() : ?int
    {
        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->means = array_fill(0, $fanIn, 0.);
        $this->variances = array_fill(0, $fanIn, 1.);
        $this->n = 0;

        $this->width = $fanOut;

        $this->beta = new Parameter(Matrix::zeros($fanIn, 1));
        $this->gamma = new Parameter(Matrix::ones($fanIn, 1));

        return $fanOut;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and
     * return an activation matrix.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if (is_null($this->n) or empty($this->means) or empty($this->variances)
            or is_null($this->beta) or is_null($this->gamma)) {
                throw new RuntimeException('Layer has not been initilaized.');
        }

        $beta = $this->beta->w()->column(0);
        $gamma = $this->gamma->w()->column(0);

        $n = $input->n();

        $oldMeans = $this->means;
        $oldVariances = $this->variances;
        $oldWeight = $this->n;

        $stdInv = $xHat = $out = [[]];

        foreach ($input as $i => $row) {
            list($mean, $variance) = Stats::meanVar($row);

            $this->means[$i] = (($n * $mean)
                + ($oldWeight * $oldMeans[$i]))
                / ($oldWeight + $n);

            $this->variances[$i] = ($oldWeight
                * $oldVariances[$i] + ($n * $variance)
                + ($oldWeight / ($n * ($oldWeight + $n)))
                * ($n * $oldMeans[$i] - $n * $mean) ** 2)
                / ($oldWeight + $n);

            $stddev = sqrt($variance + $this->epsilon);

            foreach ($row as $j => $value) {
                $a = 1. / $stddev;
                $b = $a * ($value - $mean);

                $stdInv[$i][$j] = $a;
                $xHat[$i][$j] = $b;
                $out[$i][$j] = $gamma[$i] * $b + $beta[$i];
            }
        }

        $this->stdInv = new Matrix($stdInv, false);
        $this->xHat = new Matrix($xHat, false);

        $this->n += $n;

        return new Matrix($out, false);
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (is_null($this->n) or empty($this->means) or empty($this->variances)
            or is_null($this->beta) or is_null($this->gamma)) {
                throw new RuntimeException('Layer has not been initilaized.');
        }

        $beta = $this->beta->w()->column(0);
        $gamma = $this->gamma->w()->column(0);

        $out = [[]];

        foreach ($input as $i => $row) {
            $mean = $this->means[$i];
            $variance = $this->variances[$i];
            $stddev = sqrt($variance + $this->epsilon);

            foreach ($row as $j => $value) {
                $out[$i][$j] = $gamma[$i]
                    * ($value - $mean)
                    / $stddev
                    + $beta[$i];
            }
        }

        return new Matrix($out, false);
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  callable  $prevGradient
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return callable
     */
    public function back(callable $prevGradient, Optimizer $optimizer) : callable
    {
        if (is_null($this->n) or empty($this->means) or empty($this->variances)
            or is_null($this->beta) or is_null($this->gamma)) {
                throw new RuntimeException('Layer has not been initilaized.');
        }

        if (is_null($this->stdInv) or is_null($this->xHat)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient();

        $dBeta = $dOut->sum()->asColumnMatrix();
        $dGamma = $dOut->multiply($this->xHat)->sum()->asColumnMatrix();

        $this->beta->update($optimizer->step($this->beta, $dBeta));
        $this->gamma->update($optimizer->step($this->gamma, $dGamma));

        $stdInv = $this->stdInv;
        $xHat = $this->xHat;

        unset($this->stdInv, $this->xHat);

        return function () use ($dOut, $stdInv, $xHat) {
            list($m, $n) = $dOut->shape();

            $dXHat = $dOut->multiply($this->gamma->w()->repeat(1, $n));

            $xHatSigma = $dXHat->multiply($xHat)->transpose()->sum();

            $dXHatSigma = $dXHat->transpose()->sum();

            return $dXHat->multiplyScalar($m)
                ->subtractVector($dXHatSigma)
                ->subtract($xHat->multiplyVector($xHatSigma))
                ->multiply($stdInv->divideScalar($m));
        };
    }

    /**
     * Return the parameters of the layer in an associative array.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function read() : array
    {
        if (is_null($this->n) or empty($this->means) or empty($this->variances)
            or is_null($this->beta) or is_null($this->gamma)) {
                throw new RuntimeException('Layer has not been initilaized.');
        }

        return [
            'beta' => clone $this->beta,
            'gamma' => clone $this->gamma,
        ];
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
        $this->gamma = $parameters['gamma'];
    }
}
